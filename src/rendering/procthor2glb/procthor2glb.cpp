// procthor2glb: Convert a Habitat ai2thor-hab or HSSD scene_instance.json into a
// single self-contained GLB file.
//
// Usage:
//   procthor2glb <scene_instance.json> [-o output.glb] [--normalize] [--hssd] [--hssd-lighting file.json]
//
// The tool reads the scene_instance.json, loads the stage GLB and every
// referenced object GLB, applies per-instance transforms (translation,
// rotation, scale), and writes one merged GLB.  KTX2/Basis textures are
// decoded to PNG so the output is a standard glTF 2.0 file.

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

// -- tiny_gltf (header-only glTF reader/writer) ----------------------------
// Disable STB image loading inside tinygltf — we use a custom raw loader
// for initial loading, then decode KTX2 ourselves.
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"

// -- stb_image_write (for encoding decoded pixels to PNG) ------------------
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// -- Basis Universal transcoder (KTX2/Basis → RGBA decoding) ---------------
#include "basisu_transcoder.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

enum class DatasetMode {
    AI2THOR,
    HSSD,
};

// ---------------------------------------------------------------------------
// Raw image loader: stores bytes as-is (KTX2 decoded later in the pipeline).
// ---------------------------------------------------------------------------
static bool rawImageLoader(tinygltf::Image* image,
                           const int /*imageIdx*/,
                           std::string* /*err*/,
                           std::string* /*warn*/,
                           int /*reqWidth*/, int /*reqHeight*/,
                           const unsigned char* data, int dataSize,
                           void* /*userData*/) {
    // Keep the raw compressed bytes as-is.
    image->image.assign(data, data + dataSize);
    image->width  = 0;
    image->height = 0;
    image->component  = 0;
    image->bits       = 0;
    image->pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
    return true;
}

// ---------------------------------------------------------------------------
// No-op image writer: images are already in the buffer via bufferView refs,
// so there is nothing to re-encode.
// ---------------------------------------------------------------------------
static bool noopImageWriter(const std::string* /*basepath*/,
                            const std::string* /*filename*/,
                            const tinygltf::Image* /*image*/,
                            bool /*embedImages*/,
                            const tinygltf::FsCallbacks* /*fs_cb*/,
                            const tinygltf::URICallbacks* /*uri_cb*/,
                            std::string* out_uri,
                            void* /*userData*/) {
    if (out_uri) {
        out_uri->clear();
    }
    return true;
}

// ---------------------------------------------------------------------------
// Quaternion helpers  (Habitat stores [w, x, y, z])
// ---------------------------------------------------------------------------
struct Quat {
    double w, x, y, z;
};

// Build a 4x4 column-major matrix from T * R * S
static std::array<double, 16> buildTRS(const double t[3],
                                       const Quat& q,
                                       const double s[3]) {
    double xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    double xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    double wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;

    std::array<double, 16> m{};
    m[0]  = s[0] * (1.0 - 2.0 * (yy + zz));
    m[1]  = s[0] * (2.0 * (xy + wz));
    m[2]  = s[0] * (2.0 * (xz - wy));
    m[3]  = 0.0;
    m[4]  = s[1] * (2.0 * (xy - wz));
    m[5]  = s[1] * (1.0 - 2.0 * (xx + zz));
    m[6]  = s[1] * (2.0 * (yz + wx));
    m[7]  = 0.0;
    m[8]  = s[2] * (2.0 * (xz + wy));
    m[9]  = s[2] * (2.0 * (yz - wx));
    m[10] = s[2] * (1.0 - 2.0 * (xx + yy));
    m[11] = 0.0;
    m[12] = t[0];
    m[13] = t[1];
    m[14] = t[2];
    m[15] = 1.0;
    return m;
}

// ---------------------------------------------------------------------------
// Merge a source glTF model into a destination model.
// Returns the index of a wrapper node (with optional transform) that parents
// all of the source scene's root nodes.
// ---------------------------------------------------------------------------
struct MergeResult {
    int rootNode;
};

static MergeResult mergeModel(tinygltf::Model& dst,
                              const tinygltf::Model& src,
                              const std::array<double, 16>* instanceMatrix) {
    const int bufOff   = (int)dst.buffers.size();
    const int bvOff    = (int)dst.bufferViews.size();
    const int accOff   = (int)dst.accessors.size();
    const int imgOff   = (int)dst.images.size();
    const int sampOff  = (int)dst.samplers.size();
    const int texOff   = (int)dst.textures.size();
    const int matOff   = (int)dst.materials.size();
    const int meshOff  = (int)dst.meshes.size();
    const int lightOff = (int)dst.lights.size();
    const int nodeOff  = (int)dst.nodes.size();

    // --- Buffers ---
    for (auto& b : src.buffers) dst.buffers.push_back(b);

    // --- BufferViews ---
    for (auto bv : src.bufferViews) {
        bv.buffer += bufOff;
        dst.bufferViews.push_back(bv);
    }

    // --- Accessors ---
    for (auto acc : src.accessors) {
        if (acc.bufferView >= 0) acc.bufferView += bvOff;
        if (acc.sparse.count > 0) {
            acc.sparse.indices.bufferView += bvOff;
            acc.sparse.values.bufferView  += bvOff;
        }
        dst.accessors.push_back(acc);
    }

    // --- Images ---
    for (auto img : src.images) {
        if (img.bufferView >= 0) img.bufferView += bvOff;
        dst.images.push_back(img);
    }

    // --- Samplers ---
    for (auto& s : src.samplers) dst.samplers.push_back(s);

    // --- Textures ---
    for (auto t : src.textures) {
        if (t.source >= 0) t.source += imgOff;
        if (t.sampler >= 0) t.sampler += sampOff;
        // Remap KHR_texture_basisu extension source index
        auto extIt = t.extensions.find("KHR_texture_basisu");
        if (extIt != t.extensions.end() && extIt->second.Has("source")) {
            int src_idx = extIt->second.Get("source").Get<int>();
            tinygltf::Value::Object obj;
            obj["source"] = tinygltf::Value(src_idx + imgOff);
            extIt->second = tinygltf::Value(obj);
        }
        dst.textures.push_back(t);
    }

    // --- Remap texture indices inside material TextureInfo fields ---
    auto remapTexInfo = [&](tinygltf::TextureInfo& ti) {
        if (ti.index >= 0) ti.index += texOff;
    };
    auto remapNormTexInfo = [&](tinygltf::NormalTextureInfo& ti) {
        if (ti.index >= 0) ti.index += texOff;
    };
    auto remapOccTexInfo = [&](tinygltf::OcclusionTextureInfo& ti) {
        if (ti.index >= 0) ti.index += texOff;
    };

    // --- Materials ---
    for (auto mat : src.materials) {
        remapTexInfo(mat.pbrMetallicRoughness.baseColorTexture);
        remapTexInfo(mat.pbrMetallicRoughness.metallicRoughnessTexture);
        remapNormTexInfo(mat.normalTexture);
        remapOccTexInfo(mat.occlusionTexture);
        remapTexInfo(mat.emissiveTexture);
        dst.materials.push_back(mat);
    }

    // --- Meshes ---
    for (auto mesh : src.meshes) {
        for (auto& prim : mesh.primitives) {
            for (auto& attr : prim.attributes) attr.second += accOff;
            if (prim.indices >= 0) prim.indices += accOff;
            if (prim.material >= 0) prim.material += matOff;
            for (auto& target : prim.targets)
                for (auto& attr : target) attr.second += accOff;
        }
        dst.meshes.push_back(mesh);
    }

    // --- Lights (KHR_lights_punctual) ---
    for (auto& l : src.lights) dst.lights.push_back(l);

    // --- Nodes ---
    for (auto node : src.nodes) {
        if (node.mesh >= 0) node.mesh += meshOff;
        if (node.skin >= 0) node.skin = -1;  // drop skins for now
        if (node.light >= 0) node.light += lightOff;
        for (auto& c : node.children) c += nodeOff;
        // Remap KHR_lights_punctual light index
        auto extIt = node.extensions.find("KHR_lights_punctual");
        if (extIt != node.extensions.end() && extIt->second.Has("light")) {
            int lightIdx = extIt->second.Get("light").Get<int>();
            tinygltf::Value::Object obj;
            obj["light"] = tinygltf::Value(lightIdx + lightOff);
            extIt->second = tinygltf::Value(obj);
        }
        dst.nodes.push_back(node);
    }

    // Identify source scene root nodes
    std::vector<int> srcRoots;
    if (!src.scenes.empty()) {
        int scIdx = src.defaultScene >= 0 ? src.defaultScene : 0;
        for (int n : src.scenes[scIdx].nodes)
            srcRoots.push_back(n);
    }
    if (srcRoots.empty()) {
        std::vector<bool> isChild(src.nodes.size(), false);
        for (auto& node : src.nodes)
            for (int c : node.children)
                if (c >= 0 && c < (int)isChild.size()) isChild[c] = true;
        for (int i = 0; i < (int)src.nodes.size(); i++)
            if (!isChild[i]) srcRoots.push_back(i);
    }

    // Create wrapper node with optional transform
    tinygltf::Node wrapper;
    wrapper.name = "instance";
    for (int r : srcRoots)
        wrapper.children.push_back(r + nodeOff);
    if (instanceMatrix)
        wrapper.matrix.assign(instanceMatrix->begin(), instanceMatrix->end());

    int wrapperIdx = (int)dst.nodes.size();
    dst.nodes.push_back(wrapper);
    return {wrapperIdx};
}

// ---------------------------------------------------------------------------
// Dequantize KHR_mesh_quantization: convert integer attribute accessors to
// FLOAT so the file conforms to core glTF without the extension.
// ---------------------------------------------------------------------------
static int numComponents(int type) {
    switch (type) {
        case TINYGLTF_TYPE_SCALAR: return 1;
        case TINYGLTF_TYPE_VEC2:   return 2;
        case TINYGLTF_TYPE_VEC3:   return 3;
        case TINYGLTF_TYPE_VEC4:   return 4;
        default: return 0;
    }
}

static void dequantizeMeshes(tinygltf::Model& model) {
    // Collect unique accessor indices used by mesh primitive attributes
    // (not indices — USHORT indices are already core-spec).
    std::set<int> attrAccessors;
    for (auto& mesh : model.meshes)
        for (auto& prim : mesh.primitives)
            for (auto& [name, idx] : prim.attributes)
                attrAccessors.insert(idx);

    // Build new float data in a temporary buffer
    std::vector<unsigned char> floatData;
    int dequantized = 0;

    for (int accIdx : attrAccessors) {
        auto& acc = model.accessors[accIdx];
        if (acc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) continue;
        if (acc.bufferView < 0) continue;

        auto& bv = model.bufferViews[acc.bufferView];
        auto& buf = model.buffers[bv.buffer];

        int nc = numComponents(acc.type);
        if (nc == 0) continue;

        int srcCompSize = 0;
        switch (acc.componentType) {
            case TINYGLTF_COMPONENT_TYPE_BYTE:
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  srcCompSize = 1; break;
            case TINYGLTF_COMPONENT_TYPE_SHORT:
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: srcCompSize = 2; break;
            default: continue;  // unsupported type, skip
        }
        int srcElemSize = nc * srcCompSize;
        int stride = bv.byteStride > 0 ? (int)bv.byteStride : srcElemSize;

        size_t dstOff = floatData.size();
        size_t dstElemSize = nc * sizeof(float);
        floatData.resize(dstOff + acc.count * dstElemSize);

        for (size_t e = 0; e < (size_t)acc.count; e++) {
            size_t srcOff = bv.byteOffset + acc.byteOffset + e * stride;
            float* dst = reinterpret_cast<float*>(
                &floatData[dstOff + e * dstElemSize]);

            for (int c = 0; c < nc; c++) {
                const unsigned char* src = &buf.data[srcOff + c * srcCompSize];
                float val = 0;
                switch (acc.componentType) {
                    case TINYGLTF_COMPONENT_TYPE_BYTE: {
                        int8_t v;
                        std::memcpy(&v, src, 1);
                        val = acc.normalized
                            ? std::max(v / 127.0f, -1.0f) : (float)v;
                        break;
                    }
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                        uint8_t v = *src;
                        val = acc.normalized ? v / 255.0f : (float)v;
                        break;
                    }
                    case TINYGLTF_COMPONENT_TYPE_SHORT: {
                        int16_t v;
                        std::memcpy(&v, src, 2);
                        val = acc.normalized
                            ? std::max(v / 32767.0f, -1.0f) : (float)v;
                        break;
                    }
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                        uint16_t v;
                        std::memcpy(&v, src, 2);
                        val = acc.normalized ? v / 65535.0f : (float)v;
                        break;
                    }
                    default: break;
                }
                dst[c] = val;
            }
        }

        // Update accessor min/max to float
        if (!acc.minValues.empty()) {
            for (auto& v : acc.minValues) {
                if (acc.normalized) {
                    // The min/max were stored as integer; convert to normalized float
                    // This is approximate — just clear them and let the viewer compute
                }
            }
            // For simplicity, clear min/max — viewers will recompute if needed
            // (glTF spec says they're optional except for POSITION)
            // Recompute for POSITION from the new float data
        }

        // Create new buffer view for the float data
        tinygltf::BufferView newBv;
        newBv.buffer     = -1;  // placeholder — set after buffer is added
        newBv.byteOffset = dstOff;
        newBv.byteLength = acc.count * dstElemSize;
        newBv.byteStride = 0;  // tightly packed
        newBv.target     = TINYGLTF_TARGET_ARRAY_BUFFER;

        acc.bufferView    = (int)model.bufferViews.size();
        acc.byteOffset    = 0;
        acc.componentType  = TINYGLTF_COMPONENT_TYPE_FLOAT;
        acc.normalized     = false;
        // Recompute min/max for the float data
        acc.minValues.assign(nc, std::numeric_limits<double>::max());
        acc.maxValues.assign(nc, std::numeric_limits<double>::lowest());
        for (size_t e = 0; e < (size_t)acc.count; e++) {
            const float* fp = reinterpret_cast<const float*>(
                &floatData[dstOff + e * dstElemSize]);
            for (int c = 0; c < nc; c++) {
                acc.minValues[c] = std::min(acc.minValues[c], (double)fp[c]);
                acc.maxValues[c] = std::max(acc.maxValues[c], (double)fp[c]);
            }
        }

        model.bufferViews.push_back(newBv);
        dequantized++;
    }

    if (dequantized == 0) return;

    // Add the float data as a new buffer and patch buffer view references
    int newBufIdx = (int)model.buffers.size();
    tinygltf::Buffer fb;
    fb.data = std::move(floatData);
    model.buffers.push_back(std::move(fb));

    // Patch all buffer views that have buffer == -1
    for (auto& bv : model.bufferViews) {
        if (bv.buffer == -1) bv.buffer = newBufIdx;
    }

    std::cout << "Dequantized " << dequantized
              << " attribute accessors to FLOAT\n";
}

// ---------------------------------------------------------------------------
// Promote KHR_texture_basisu extension source to standard texture.source,
// so the file works without the extension.
// ---------------------------------------------------------------------------
static void promoteBasisTextures(tinygltf::Model& model) {
    int promoted = 0;
    for (auto& tex : model.textures) {
        auto it = tex.extensions.find("KHR_texture_basisu");
        if (it == tex.extensions.end()) continue;

        // The extension value is {"source": <int>}
        if (it->second.Has("source")) {
            int src = it->second.Get("source").Get<int>();
            if (tex.source < 0) {
                tex.source = src;  // promote to standard field
            }
        }
        tex.extensions.erase(it);
        promoted++;
    }
    if (promoted > 0)
        std::cout << "Promoted " << promoted
                  << " KHR_texture_basisu textures to standard source\n";
}

// ---------------------------------------------------------------------------
// Decode KTX2/Basis images to PNG so the GLB is a standard glTF 2.0 file.
// ---------------------------------------------------------------------------
static void stbiWriteToVec(void* ctx, void* data, int size) {
    auto* v = static_cast<std::vector<unsigned char>*>(ctx);
    auto* bytes = static_cast<const unsigned char*>(data);
    v->insert(v->end(), bytes, bytes + size);
}

static void decodeKTX2Images(tinygltf::Model& model) {
    static bool inited = false;
    if (!inited) {
        basist::basisu_transcoder_init();
        inited = true;
    }

    int decoded = 0;
    for (size_t imgIdx = 0; imgIdx < model.images.size(); imgIdx++) {
        auto& img = model.images[imgIdx];

        // Get image data (from bufferView or image.image)
        const unsigned char* data = nullptr;
        size_t dataSize = 0;

        if (img.bufferView >= 0) {
            auto& bv = model.bufferViews[img.bufferView];
            auto& buf = model.buffers[bv.buffer];
            data = buf.data.data() + bv.byteOffset;
            dataSize = bv.byteLength;
        } else if (!img.image.empty()) {
            data = img.image.data();
            dataSize = img.image.size();
        }

        if (!data || dataSize < 12) continue;

        // Check KTX2 magic: «KTX 20»\r\n\x1a\n
        static const uint8_t ktx2Magic[12] = {
            0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB,
            0x0D, 0x0A, 0x1A, 0x0A
        };
        bool isKTX2 = (img.mimeType == "image/ktx2") ||
                      (std::memcmp(data, ktx2Magic, 12) == 0);
        if (!isKTX2) continue;

        // Transcode KTX2 → RGBA32
        basist::ktx2_transcoder transcoder;
        if (!transcoder.init(data, static_cast<uint32_t>(dataSize))) {
            std::cerr << "  warn: failed to init KTX2 transcoder for image "
                      << imgIdx << "\n";
            continue;
        }

        if (!transcoder.start_transcoding()) {
            std::cerr << "  warn: failed to start KTX2 transcoding for image "
                      << imgIdx << "\n";
            continue;
        }

        basist::ktx2_image_level_info levelInfo;
        if (!transcoder.get_image_level_info(levelInfo, 0, 0, 0)) {
            std::cerr << "  warn: failed to get KTX2 level info for image "
                      << imgIdx << "\n";
            continue;
        }

        uint32_t w = levelInfo.m_orig_width;
        uint32_t h = levelInfo.m_orig_height;
        std::vector<uint8_t> rgba(w * h * 4);

        if (!transcoder.transcode_image_level(
                0, 0, 0,  // level, layer, face
                rgba.data(), static_cast<uint32_t>(w * h),
                basist::transcoder_texture_format::cTFRGBA32)) {
            std::cerr << "  warn: failed to transcode KTX2 image "
                      << imgIdx << "\n";
            continue;
        }

        // Encode as PNG into memory
        std::vector<unsigned char> pngData;
        pngData.reserve(w * h);  // rough estimate
        stbi_write_png_to_func(stbiWriteToVec, &pngData,
                               static_cast<int>(w), static_cast<int>(h),
                               4, rgba.data(),
                               static_cast<int>(w * 4));
        if (pngData.empty()) {
            std::cerr << "  warn: PNG encoding failed for image "
                      << imgIdx << "\n";
            continue;
        }

        // Store PNG data in a new buffer + bufferView
        int newBufIdx = static_cast<int>(model.buffers.size());
        tinygltf::Buffer newBuf;
        newBuf.data = std::move(pngData);
        model.buffers.push_back(std::move(newBuf));

        int newBvIdx = static_cast<int>(model.bufferViews.size());
        tinygltf::BufferView newBv;
        newBv.buffer     = newBufIdx;
        newBv.byteOffset = 0;
        newBv.byteLength = model.buffers.back().data.size();
        model.bufferViews.push_back(newBv);

        // Update image to reference the new PNG data
        img.bufferView = newBvIdx;
        img.mimeType   = "image/png";
        img.width      = static_cast<int>(w);
        img.height     = static_cast<int>(h);
        img.component  = 4;
        img.bits       = 8;
        img.pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
        img.image.clear();

        decoded++;
    }

    if (decoded > 0)
        std::cout << "Decoded " << decoded << " KTX2 textures to PNG\n";
}

// ---------------------------------------------------------------------------
// Bake KHR_texture_transform into UV coordinates so the extension can be
// removed entirely.  For each primitive whose material uses the extension,
// we create new TEXCOORD accessors with the offset/scale/rotation applied.
// ---------------------------------------------------------------------------
struct UVTransform {
    double offset[2] = {0, 0};
    double scale[2]  = {1, 1};
    double rotation  = 0;
    bool operator==(const UVTransform& o) const {
        return offset[0] == o.offset[0] && offset[1] == o.offset[1] &&
               scale[0] == o.scale[0] && scale[1] == o.scale[1] &&
               rotation == o.rotation;
    }
};

static UVTransform extractUVTransform(const tinygltf::Value& ext) {
    UVTransform t;
    if (ext.Has("offset")) {
        auto& o = ext.Get("offset");
        if (o.ArrayLen() >= 2) {
            t.offset[0] = o.Get(0).GetNumberAsDouble();
            t.offset[1] = o.Get(1).GetNumberAsDouble();
        }
    }
    if (ext.Has("scale")) {
        auto& s = ext.Get("scale");
        if (s.ArrayLen() >= 2) {
            t.scale[0] = s.Get(0).GetNumberAsDouble();
            t.scale[1] = s.Get(1).GetNumberAsDouble();
        }
    }
    if (ext.Has("rotation"))
        t.rotation = ext.Get("rotation").GetNumberAsDouble();
    return t;
}

// Create a new VEC2 FLOAT accessor with the UV transform baked in.
static int createTransformedUVAccessor(tinygltf::Model& model,
                                       int srcAccIdx,
                                       const UVTransform& t) {
    auto& srcAcc = model.accessors[srcAccIdx];
    if (srcAcc.type != TINYGLTF_TYPE_VEC2 ||
        srcAcc.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT ||
        srcAcc.bufferView < 0)
        return srcAccIdx;

    auto& srcBv  = model.bufferViews[srcAcc.bufferView];
    auto& srcBuf = model.buffers[srcBv.buffer];
    int stride = srcBv.byteStride > 0
                     ? (int)srcBv.byteStride
                     : (int)(2 * sizeof(float));

    double cosR = std::cos(t.rotation);
    double sinR = std::sin(t.rotation);

    size_t count = srcAcc.count;
    std::vector<unsigned char> newData(count * 2 * sizeof(float));
    double minU = 1e30, minV = 1e30, maxU = -1e30, maxV = -1e30;

    for (size_t i = 0; i < count; i++) {
        size_t off = srcBv.byteOffset + srcAcc.byteOffset + i * stride;
        float u, v;
        std::memcpy(&u, &srcBuf.data[off],                sizeof(float));
        std::memcpy(&v, &srcBuf.data[off + sizeof(float)], sizeof(float));

        // KHR_texture_transform spec:
        //   uv' = R(θ) * (uv * scale) + offset
        double su = u * t.scale[0];
        double sv = v * t.scale[1];
        double nu =  su * cosR + sv * sinR + t.offset[0];
        double nv = -su * sinR + sv * cosR + t.offset[1];

        float fu = (float)nu, fv = (float)nv;
        std::memcpy(&newData[(i * 2)     * sizeof(float)], &fu, sizeof(float));
        std::memcpy(&newData[(i * 2 + 1) * sizeof(float)], &fv, sizeof(float));

        minU = std::min(minU, nu); maxU = std::max(maxU, nu);
        minV = std::min(minV, nv); maxV = std::max(maxV, nv);
    }

    int newBufIdx = (int)model.buffers.size();
    tinygltf::Buffer buf;
    buf.data = std::move(newData);
    model.buffers.push_back(std::move(buf));

    int newBvIdx = (int)model.bufferViews.size();
    tinygltf::BufferView bv;
    bv.buffer     = newBufIdx;
    bv.byteOffset = 0;
    bv.byteLength = count * 2 * sizeof(float);
    bv.target     = TINYGLTF_TARGET_ARRAY_BUFFER;
    model.bufferViews.push_back(bv);

    int newAccIdx = (int)model.accessors.size();
    tinygltf::Accessor acc;
    acc.bufferView    = newBvIdx;
    acc.byteOffset    = 0;
    acc.componentType  = TINYGLTF_COMPONENT_TYPE_FLOAT;
    acc.count          = count;
    acc.type           = TINYGLTF_TYPE_VEC2;
    acc.minValues      = {minU, minV};
    acc.maxValues      = {maxU, maxV};
    model.accessors.push_back(acc);

    return newAccIdx;
}

static void bakeTextureTransforms(tinygltf::Model& model) {
    // Map (srcAccessorIdx, transform) → new accessor idx, to deduplicate.
    struct CacheKey {
        int accIdx;
        double o0, o1, s0, s1, r;
        bool operator<(const CacheKey& k) const {
            return std::tie(accIdx, o0, o1, s0, s1, r) <
                   std::tie(k.accIdx, k.o0, k.o1, k.s0, k.s1, k.r);
        }
    };
    std::map<CacheKey, int> accCache;

    auto getCached = [&](int srcAcc, const UVTransform& t) -> int {
        CacheKey key{srcAcc, t.offset[0], t.offset[1],
                     t.scale[0], t.scale[1], t.rotation};
        auto it = accCache.find(key);
        if (it != accCache.end()) return it->second;
        int r = createTransformedUVAccessor(model, srcAcc, t);
        accCache[key] = r;
        return r;
    };

    // --- Phase 1: group primitives by material index ---
    struct PrimRef { int meshIdx, primIdx; };
    std::map<int, std::vector<PrimRef>> matPrims;
    for (int mi = 0; mi < (int)model.meshes.size(); mi++)
        for (int pi = 0; pi < (int)model.meshes[mi].primitives.size(); pi++) {
            int m = model.meshes[mi].primitives[pi].material;
            if (m >= 0) matPrims[m].push_back({mi, pi});
        }

    int bakedCount = 0;

    // --- Phase 2: for each material with transforms, bake UVs ---
    for (auto& [matIdx, prims] : matPrims) {
        auto& mat = model.materials[matIdx];

        // Gather all (texCoord, transform?) from every texture slot
        struct SlotRef {
            int texCoord;
            UVTransform transform;
            bool hasTransform;
        };
        std::vector<SlotRef> slots;

        auto gatherSlot = [&](auto& texInfo) {
            if (texInfo.index < 0) return;
            SlotRef s;
            s.texCoord = texInfo.texCoord;
            auto it = texInfo.extensions.find("KHR_texture_transform");
            s.hasTransform = (it != texInfo.extensions.end());
            if (s.hasTransform)
                s.transform = extractUVTransform(it->second);
            slots.push_back(s);
        };
        gatherSlot(mat.pbrMetallicRoughness.baseColorTexture);
        gatherSlot(mat.pbrMetallicRoughness.metallicRoughnessTexture);
        gatherSlot(mat.normalTexture);
        gatherSlot(mat.occlusionTexture);
        gatherSlot(mat.emissiveTexture);

        bool anyTransform = false;
        for (auto& s : slots)
            if (s.hasTransform) { anyTransform = true; break; }
        if (!anyTransform) continue;

        // Check if untransformed textures also use texCoord 0
        bool untransformedOnTC0 = false;
        UVTransform commonTransform{};
        bool firstTx = true;
        bool allSame = true;
        for (auto& s : slots) {
            if (!s.hasTransform && s.texCoord == 0)
                untransformedOnTC0 = true;
            if (s.hasTransform) {
                if (firstTx) { commonTransform = s.transform; firstTx = false; }
                else if (!(s.transform == commonTransform)) allSame = false;
            }
        }

        if (allSame && !untransformedOnTC0) {
            // Simple: all transformed textures share the same transform and
            // no untransformed textures compete for TEXCOORD_0.
            // → bake into TEXCOORD_0 in-place.
            for (auto& pr : prims) {
                auto& prim = model.meshes[pr.meshIdx].primitives[pr.primIdx];
                auto tcIt = prim.attributes.find("TEXCOORD_0");
                if (tcIt == prim.attributes.end()) continue;
                tcIt->second = getCached(tcIt->second, commonTransform);
            }
        } else {
            // Mixed: some textures are untransformed on the same texCoord,
            // or there are multiple distinct transforms.
            // → keep TEXCOORD_0 untouched; create TEXCOORD_N for each
            //   unique transform and redirect the texture infos.

            // Collect unique transforms
            std::vector<UVTransform> uniq;
            for (auto& s : slots) {
                if (!s.hasTransform) continue;
                bool found = false;
                for (auto& u : uniq)
                    if (u == s.transform) { found = true; break; }
                if (!found) uniq.push_back(s.transform);
            }

            for (auto& pr : prims) {
                auto& prim = model.meshes[pr.meshIdx].primitives[pr.primIdx];
                auto tcIt = prim.attributes.find("TEXCOORD_0");
                if (tcIt == prim.attributes.end()) continue;
                int srcAcc = tcIt->second;

                // Find next free TEXCOORD index
                int nextTC = 0;
                for (auto& [name, _] : prim.attributes)
                    if (name.rfind("TEXCOORD_", 0) == 0) {
                        int n = std::stoi(name.substr(9));
                        nextTC = std::max(nextTC, n + 1);
                    }

                for (size_t ti = 0; ti < uniq.size(); ti++) {
                    int newAcc = getCached(srcAcc, uniq[ti]);
                    prim.attributes["TEXCOORD_" + std::to_string(nextTC + (int)ti)]
                        = newAcc;
                }
            }

            // Update material texture infos: redirect transformed ones
            if (!prims.empty()) {
                auto& firstPrim =
                    model.meshes[prims[0].meshIdx]
                        .primitives[prims[0].primIdx];
                int nextTC = 0;
                for (auto& [name, _] : firstPrim.attributes)
                    if (name.rfind("TEXCOORD_", 0) == 0) {
                        int n = std::stoi(name.substr(9));
                        nextTC = std::max(nextTC, n + 1);
                    }
                int baseTC = nextTC - (int)uniq.size();

                auto redirect = [&](auto& texInfo) {
                    if (texInfo.index < 0) return;
                    auto it = texInfo.extensions.find("KHR_texture_transform");
                    if (it == texInfo.extensions.end()) return;
                    UVTransform t = extractUVTransform(it->second);
                    for (size_t ti = 0; ti < uniq.size(); ti++) {
                        if (uniq[ti] == t) {
                            texInfo.texCoord = baseTC + (int)ti;
                            break;
                        }
                    }
                    texInfo.extensions.erase("KHR_texture_transform");
                };
                redirect(mat.pbrMetallicRoughness.baseColorTexture);
                redirect(mat.pbrMetallicRoughness.metallicRoughnessTexture);
                redirect(mat.normalTexture);
                redirect(mat.occlusionTexture);
                redirect(mat.emissiveTexture);
            }

            bakedCount += (int)prims.size();
            continue;  // skip the simple-path cleanup below
        }

        // Simple-path: remove extensions from material texture infos
        auto removeTT = [](auto& texInfo) {
            texInfo.extensions.erase("KHR_texture_transform");
        };
        removeTT(mat.pbrMetallicRoughness.baseColorTexture);
        removeTT(mat.pbrMetallicRoughness.metallicRoughnessTexture);
        removeTT(mat.normalTexture);
        removeTT(mat.occlusionTexture);
        removeTT(mat.emissiveTexture);
        bakedCount += (int)prims.size();
    }

    if (bakedCount > 0)
        std::cout << "Baked KHR_texture_transform into UVs for "
                  << bakedCount << " primitives\n";
}

// ---------------------------------------------------------------------------
// Strip specific extensions from extensionsUsed / extensionsRequired.
// ---------------------------------------------------------------------------
static void stripExtensions(tinygltf::Model& model,
                            const std::vector<std::string>& toRemove) {
    auto remove = [&](std::vector<std::string>& list) {
        list.erase(std::remove_if(list.begin(), list.end(),
            [&](const std::string& s) {
                return std::find(toRemove.begin(), toRemove.end(), s) !=
                       toRemove.end();
            }),
            list.end());
    };
    remove(model.extensionsUsed);
    remove(model.extensionsRequired);
}

// ---------------------------------------------------------------------------
// Consolidate all buffers into buffer 0 (GLB binary chunk only stores buf 0).
// ---------------------------------------------------------------------------
static void consolidateBuffers(tinygltf::Model& model) {
    if (model.buffers.size() <= 1) {
        if (!model.buffers.empty()) model.buffers[0].uri.clear();
        return;
    }

    std::cout << "Consolidating " << model.buffers.size()
              << " buffers into one...\n";

    std::vector<size_t> offset(model.buffers.size());
    size_t total = 0;
    for (size_t i = 0; i < model.buffers.size(); i++) {
        offset[i] = total;
        total += model.buffers[i].data.size();
    }

    std::vector<unsigned char> merged;
    merged.reserve(total);
    for (auto& buf : model.buffers)
        merged.insert(merged.end(), buf.data.begin(), buf.data.end());

    for (auto& bv : model.bufferViews) {
        if (bv.buffer >= 0 && bv.buffer < (int)offset.size()) {
            bv.byteOffset += offset[bv.buffer];
            bv.buffer = 0;
        }
    }

    model.buffers.clear();
    tinygltf::Buffer buf;
    buf.data = std::move(merged);
    model.buffers.push_back(std::move(buf));
}

// ---------------------------------------------------------------------------
// Collect all extensionsUsed / extensionsRequired from source models
// ---------------------------------------------------------------------------
static void mergeExtensions(tinygltf::Model& dst,
                            const tinygltf::Model& src) {
    auto addUnique = [](std::vector<std::string>& dst,
                        const std::vector<std::string>& src) {
        for (auto& s : src) {
            if (std::find(dst.begin(), dst.end(), s) == dst.end())
                dst.push_back(s);
        }
    };
    addUnique(dst.extensionsUsed, src.extensionsUsed);
    addUnique(dst.extensionsRequired, src.extensionsRequired);
}

// ---------------------------------------------------------------------------
// Cache for loaded glTF models (avoid re-loading the same object GLB)
// ---------------------------------------------------------------------------
struct GltfCache {
    std::map<std::string, tinygltf::Model> models;
    std::map<std::string, bool> failedPaths;  // remember failures too
    tinygltf::TinyGLTF loader;

    GltfCache() {
        loader.SetImageLoader(rawImageLoader, nullptr);
    }

    const tinygltf::Model* load(const std::string& path) {
        if (failedPaths.count(path)) return nullptr;
        auto it = models.find(path);
        if (it != models.end()) return &it->second;

        tinygltf::Model model;
        std::string err, warn;
        bool ok;
        if (path.size() >= 4 && path.substr(path.size() - 4) == ".glb") {
            ok = loader.LoadBinaryFromFile(&model, &err, &warn, path);
        } else {
            ok = loader.LoadASCIIFromFile(&model, &err, &warn, path);
        }
        if (!warn.empty())
            std::cerr << "  warn [" << fs::path(path).filename().string()
                      << "]: " << warn << "\n";
        if (!err.empty())
            std::cerr << "  err  [" << fs::path(path).filename().string()
                      << "]: " << err << "\n";
        if (!ok) {
            failedPaths[path] = true;
            return nullptr;
        }

        auto [iter, _] = models.emplace(path, std::move(model));
        return &iter->second;
    }
};

// ---------------------------------------------------------------------------
// Compute a rotation matrix that corrects an asset's front/up orientation
// to match the glTF/Habitat world convention (front = -Z, up = +Y).
//
// Given an asset whose "front" direction is `f` and "up" is `u`, we build
// the rotation R such that:
//   R * f  → [0, 0, -1]   (world front)
//   R * u  → [0, 1,  0]   (world up)
//   R * r  → [1, 0,  0]   (world right,  r = f × u)
// ---------------------------------------------------------------------------
static std::array<double, 16> buildFrontUpCorrection(const double f[3],
                                                     const double u[3]) {
    // asset_right = front × up  (cross product)
    double rx = f[1] * u[2] - f[2] * u[1];
    double ry = f[2] * u[0] - f[0] * u[2];
    double rz = f[0] * u[1] - f[1] * u[0];
    // normalize
    double rlen = std::sqrt(rx * rx + ry * ry + rz * rz);
    if (rlen < 1e-9) {
        // Degenerate — return identity
        return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    }
    rx /= rlen; ry /= rlen; rz /= rlen;

    // B = [right | up | -front] as columns (orthonormal basis of asset frame
    //      expressed in the same coordinate system as front/up)
    // R = B^T  (orthogonal inverse)
    //
    // Column-major 4×4:
    //   col0 = row0 of B = [rx, ux, -fx]
    //   col1 = row1 of B = [ry, uy, -fy]
    //   col2 = row2 of B = [rz, uz, -fz]
    std::array<double, 16> m{};
    // col0
    m[0]  = rx;  m[1]  = u[0]; m[2]  = -f[0]; m[3]  = 0;
    // col1
    m[4]  = ry;  m[5]  = u[1]; m[6]  = -f[1]; m[7]  = 0;
    // col2
    m[8]  = rz;  m[9]  = u[2]; m[10] = -f[2]; m[11] = 0;
    // col3
    m[12] = 0;   m[13] = 0;    m[14] = 0;     m[15] = 1;
    return m;
}

static std::string lowerString(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

static tinygltf::Value jsonToTinyValue(const json& value) {
    if (value.is_boolean()) {
        return tinygltf::Value(value.get<bool>());
    }
    if (value.is_number()) {
        return tinygltf::Value(value.get<double>());
    }
    if (value.is_string()) {
        return tinygltf::Value(value.get<std::string>());
    }
    if (value.is_array()) {
        tinygltf::Value::Array array;
        for (const auto& item : value) {
            array.push_back(jsonToTinyValue(item));
        }
        return tinygltf::Value(std::move(array));
    }
    if (value.is_object()) {
        tinygltf::Value::Object object;
        for (auto it = value.begin(); it != value.end(); ++it) {
            object[it.key()] = jsonToTinyValue(it.value());
        }
        return tinygltf::Value(std::move(object));
    }
    return tinygltf::Value();
}

static bool jsonVec3(const json& value, double out[3]) {
    if (!value.is_array() || value.size() < 3) return false;
    for (int i = 0; i < 3; i++) {
        if (!value[i].is_number()) return false;
        out[i] = value[i].get<double>();
    }
    return true;
}

static bool readVec3ByKeys(const json& object,
                           const std::vector<std::string>& keys,
                           double out[3]) {
    for (const auto& key : keys) {
        if (object.contains(key) && jsonVec3(object[key], out)) {
            return true;
        }
    }
    return false;
}

static bool readNumberByKeys(const json& object,
                             const std::vector<std::string>& keys,
                             double& out) {
    for (const auto& key : keys) {
        if (object.contains(key) && object[key].is_number()) {
            out = object[key].get<double>();
            return true;
        }
    }
    return false;
}

static bool readStringByKeys(const json& object,
                             const std::vector<std::string>& keys,
                             std::string& out) {
    for (const auto& key : keys) {
        if (object.contains(key) && object[key].is_string()) {
            out = object[key].get<std::string>();
            return true;
        }
    }
    return false;
}

static std::array<double, 16> buildLightDirectionMatrix(const double t[3],
                                                       const double direction[3]) {
    double dir[3] = {direction[0], direction[1], direction[2]};
    double len = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    if (len < 1e-9) {
        return {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            t[0], t[1], t[2], 1
        };
    }
    for (double& v : dir) v /= len;

    double z[3] = {-dir[0], -dir[1], -dir[2]};
    double up[3] = {0, 1, 0};
    double x[3] = {
        up[1] * z[2] - up[2] * z[1],
        up[2] * z[0] - up[0] * z[2],
        up[0] * z[1] - up[1] * z[0]
    };
    double x_len = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    if (x_len < 1e-9) {
        up[0] = 1;
        up[1] = 0;
        up[2] = 0;
        x[0] = up[1] * z[2] - up[2] * z[1];
        x[1] = up[2] * z[0] - up[0] * z[2];
        x[2] = up[0] * z[1] - up[1] * z[0];
        x_len = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    }
    for (double& v : x) v /= x_len;
    double y[3] = {
        z[1] * x[2] - z[2] * x[1],
        z[2] * x[0] - z[0] * x[2],
        z[0] * x[1] - z[1] * x[0]
    };

    return {
        x[0], x[1], x[2], 0,
        y[0], y[1], y[2], 0,
        z[0], z[1], z[2], 0,
        t[0], t[1], t[2], 1
    };
}

// ---------------------------------------------------------------------------
// Resolve dataset-specific asset and config paths.
// ---------------------------------------------------------------------------
static std::string resolveStageAssetPath(DatasetMode mode,
                                         const fs::path& datasetRoot,
                                         const std::string& templateName) {
    if (mode == DatasetMode::HSSD) {
        fs::path relative(templateName + ".glb");
        if (relative.has_parent_path()) {
            return (datasetRoot / relative).string();
        }
        return (datasetRoot / "stages" / (templateName + ".glb")).string();
    }
    return (datasetRoot / "assets" / (templateName + ".glb")).string();
}

static std::string resolveStageConfigPath(DatasetMode mode,
                                          const fs::path& datasetRoot,
                                          const std::string& templateName) {
    if (mode == DatasetMode::HSSD) {
        fs::path relative(templateName + ".stage_config.json");
        if (relative.has_parent_path()) {
            return (datasetRoot / relative).string();
        }
        return (datasetRoot / "stages" / (templateName + ".stage_config.json")).string();
    }
    return (datasetRoot / "configs" / (templateName + ".stage_config.json")).string();
}

static std::string resolveObjectAssetPath(DatasetMode mode,
                                          const fs::path& datasetRoot,
                                          const std::string& templateName) {
    if (mode == DatasetMode::HSSD) {
        fs::path relative(templateName + ".glb");
        if (relative.has_parent_path()) {
            return (datasetRoot / relative).string();
        }
        if (!templateName.empty()) {
            fs::path path = datasetRoot / "objects" / templateName.substr(0, 1) / (templateName + ".glb");
            if (fs::exists(path)) return path.string();
            path = datasetRoot / "objects" / "openings" / (templateName + ".glb");
            if (fs::exists(path)) return path.string();
            const std::string partMarker = "_part_";
            const size_t partPos = templateName.find(partMarker);
            if (partPos != std::string::npos) {
                const std::string baseTemplate = templateName.substr(0, partPos);
                path = datasetRoot / "objects" / "decomposed" / baseTemplate / (templateName + ".glb");
                if (fs::exists(path)) return path.string();
            }
            return (datasetRoot / "objects" / templateName.substr(0, 1) / (templateName + ".glb")).string();
        }
    }
    return (datasetRoot / "assets" / (templateName + ".glb")).string();
}

static std::string resolveObjectConfigPath(DatasetMode mode,
                                           const fs::path& datasetRoot,
                                           const std::string& templateName) {
    if (mode == DatasetMode::HSSD) {
        fs::path relative(templateName + ".object_config.json");
        if (relative.has_parent_path()) {
            return (datasetRoot / relative).string();
        }
        if (!templateName.empty()) {
            fs::path path = datasetRoot / "objects" / templateName.substr(0, 1) / (templateName + ".object_config.json");
            if (fs::exists(path)) return path.string();
            path = datasetRoot / "objects" / "openings" / (templateName + ".object_config.json");
            if (fs::exists(path)) return path.string();
            const std::string partMarker = "_part_";
            const size_t partPos = templateName.find(partMarker);
            if (partPos != std::string::npos) {
                const std::string baseTemplate = templateName.substr(0, partPos);
                path = datasetRoot / "objects" / "decomposed" / baseTemplate / (templateName + ".object_config.json");
                if (fs::exists(path)) return path.string();
            }
            return (datasetRoot / "objects" / templateName.substr(0, 1) / (templateName + ".object_config.json")).string();
        }
    }
    return (datasetRoot / "configs" / (templateName + ".object_config.json")).string();
}

// ---------------------------------------------------------------------------
// Find dataset root.
// ---------------------------------------------------------------------------
static fs::path findDatasetRoot(const fs::path& sceneJsonPath,
                                DatasetMode mode) {
    fs::path dir = fs::canonical(sceneJsonPath).parent_path();
    for (int i = 0; i < 10; i++) {
        if (mode == DatasetMode::HSSD) {
            if (fs::exists(dir / "stages") && fs::exists(dir / "objects"))
                return dir;
        } else if (fs::exists(dir / "assets") && fs::exists(dir / "configs")) {
            return dir;
        }
        if (!dir.has_parent_path() || dir == dir.parent_path()) break;
        dir = dir.parent_path();
    }
    return fs::canonical(sceneJsonPath).parent_path().parent_path().parent_path();
}

static void readScale(const json& instance, double s[3]) {
    auto readScaleValue = [&](const json& value) {
        if (value.is_number()) {
            const double scale = value.get<double>();
            s[0] = scale;
            s[1] = scale;
            s[2] = scale;
            return true;
        }
        if (value.is_array() && value.size() >= 3) {
            for (int i = 0; i < 3; i++) {
                if (!value[i].is_number()) return false;
                s[i] = value[i].get<double>();
            }
            return true;
        }
        return false;
    };
    if (instance.count("non_uniform_scale") && readScaleValue(instance["non_uniform_scale"])) {
        return;
    }
    if (instance.count("scale") && readScaleValue(instance["scale"])) {
        return;
    }
    if (instance.count("uniform_scale")) {
        readScaleValue(instance["uniform_scale"]);
    }
}

struct HssdLightingImport {
    int punctual = 0;
    int unsupported = 0;
    int environment = 0;
    int missing_sources = 0;
    bool found_source = false;
};

struct HssdLightingSource {
    std::string label;
    json value;
    double positive_intensity_scale = 1.0;
    double negative_intensity_scale = 1.0;
};

static HssdLightingSource habitatDefaultLightingSource() {
    HssdLightingSource source;
    source.label = "default_lighting:habitat_default";
    source.value = {
        {"lights", {
            {"0", {
                {"name", "habitat_default_minus_z"},
                {"type", "directional"},
                {"direction", {0.0, -0.5, -0.5}},
                {"color", {0.5, 0.5, 0.5}}
            }},
            {"1", {
                {"name", "habitat_default_plus_z"},
                {"type", "directional"},
                {"direction", {0.0, -0.5, 0.5}},
                {"color", {0.5, 0.5, 0.5}}
            }},
            {"2", {
                {"name", "habitat_default_minus_x"},
                {"type", "directional"},
                {"direction", {-0.5, -0.5, 0.0}},
                {"color", {0.5, 0.5, 0.5}}
            }},
            {"3", {
                {"name", "habitat_default_plus_x"},
                {"type", "directional"},
                {"direction", {0.5, -0.5, 0.0}},
                {"color", {0.5, 0.5, 0.5}}
            }}
        }}
    };
    return source;
}

static fs::path resolveHssdReferencePath(const fs::path& datasetRoot,
                                         const std::string& path) {
    fs::path p(path);
    if (p.is_absolute()) return p;
    std::vector<fs::path> candidates;
    candidates.push_back(datasetRoot / p);
    if (!p.has_extension()) {
        const fs::path lightingFile(path + ".lighting_config.json");
        candidates.push_back(datasetRoot / lightingFile);
        candidates.push_back(datasetRoot / "lights" / lightingFile);
        candidates.push_back(datasetRoot / "lighting" / lightingFile);
        candidates.push_back(datasetRoot / "data" / lightingFile);
    }
    for (const auto& candidate : candidates) {
        if (fs::exists(candidate)) return candidate;
    }
    return candidates.front();
}

static bool readJsonFile(const fs::path& path, json& out) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    try {
        out = json::parse(file);
    } catch (const json::exception&) {
        return false;
    }
    return true;
}

static std::string lightTypeFromRecord(const json& record) {
    std::string type;
    const bool hasType = readStringByKeys(record, {"type", "light_type", "lightType"}, type);
    type = lowerString(type);
    if (type.find("spot") != std::string::npos) return "spot";
    if (type.find("point") != std::string::npos) return "point";
    if (type.find("directional") != std::string::npos ||
        type.find("direct") != std::string::npos ||
        type.find("sun") != std::string::npos) {
        return "directional";
    }
    if (hasType && !type.empty()) return "";
    if (record.contains("vector") && record["vector"].is_array() && record["vector"].size() >= 4 &&
        record["vector"][3].is_number()) {
        return std::abs(record["vector"][3].get<double>()) < 1e-9 ? "directional" : "point";
    }
    if (record.contains("position") && record["position"].is_array() && record["position"].size() >= 4 &&
        record["position"][3].is_number()) {
        return std::abs(record["position"][3].get<double>()) < 1e-9 ? "directional" : "point";
    }
    if (record.contains("direction") || record.contains("normal")) return "directional";
    if (record.contains("position") || record.contains("translation")) return "point";
    return "";
}

static bool isEnvironmentLightingRecord(const json& record) {
    if (!record.is_object()) return false;
    for (const auto& key : {"ambient", "environment", "envmap", "ibl", "sky", "skydome", "background"}) {
        if (record.contains(key)) return true;
    }
    std::string type;
    if (readStringByKeys(record, {"type", "light_type", "lightType"}, type)) {
        type = lowerString(type);
        return type.find("ambient") != std::string::npos ||
               type.find("environment") != std::string::npos ||
               type.find("ibl") != std::string::npos ||
               type.find("sky") != std::string::npos;
    }
    return false;
}

static bool isLightLikeRecord(const json& record) {
    if (!record.is_object()) return false;
    if (!lightTypeFromRecord(record).empty()) return true;
    if (isEnvironmentLightingRecord(record)) return true;
    return (record.contains("color") || record.contains("intensity")) &&
           (record.contains("position") || record.contains("translation") ||
            record.contains("direction") || record.contains("vector"));
}

static void collectHssdLightRecords(const json& value,
                                    std::vector<json>& records) {
    if (value.is_array()) {
        for (const auto& item : value) {
            collectHssdLightRecords(item, records);
        }
        return;
    }
    if (!value.is_object()) return;
    bool foundContainer = false;
    for (const auto& key : {"lights", "light_setup", "lightSetup", "light_setups", "lightSetups", "lighting"}) {
        if (value.contains(key)) {
            collectHssdLightRecords(value[key], records);
            foundContainer = true;
        }
    }
    if (foundContainer) return;
    bool foundMappedLight = false;
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (it.value().is_object() && isLightLikeRecord(it.value())) {
            collectHssdLightRecords(it.value(), records);
            foundMappedLight = true;
        }
    }
    if (foundMappedLight) return;
    if (isLightLikeRecord(value)) {
        records.push_back(value);
        return;
    }
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (it.value().is_object() || it.value().is_array()) {
            collectHssdLightRecords(it.value(), records);
        }
    }
}

static void addRawHssdLightingExtras(tinygltf::Value::Object& hssdLighting,
                                     const std::string& key,
                                     const json& value) {
    if (!value.is_null() && !((value.is_object() || value.is_array()) && value.empty())) {
        hssdLighting[key] = jsonToTinyValue(value);
    }
}

static bool readLightVector(const json& record,
                            const std::string& type,
                            double position[3],
                            double direction[3],
                            bool& has_position,
                            bool& has_direction) {
    has_position = readVec3ByKeys(record, {"position", "translation"}, position);
    has_direction = readVec3ByKeys(record, {"direction", "normal"}, direction);
    if (record.contains("vector") && record["vector"].is_array() && record["vector"].size() >= 3) {
        double v[3] = {0, 0, -1};
        if (jsonVec3(record["vector"], v)) {
            const bool directional_vector = record["vector"].size() >= 4 &&
                record["vector"][3].is_number() &&
                std::abs(record["vector"][3].get<double>()) < 1e-9;
            if (directional_vector || type == "directional") {
                direction[0] = v[0];
                direction[1] = v[1];
                direction[2] = v[2];
                has_direction = true;
            } else if (!has_position) {
                position[0] = v[0];
                position[1] = v[1];
                position[2] = v[2];
                has_position = true;
            }
        }
    }
    return has_position || has_direction;
}

static void addHssdUnsupportedLightNode(tinygltf::Model& model,
                                        tinygltf::Scene& scene,
                                        const json& record,
                                        const std::string& source,
                                        const std::string& reason) {
    tinygltf::Node node;
    node.name = "hssd_unsupported_light_" + std::to_string(model.nodes.size());
    double t[3] = {0, 0, 0};
    if (record.is_object() && readVec3ByKeys(record, {"position", "translation"}, t)) {
        node.translation = {t[0], t[1], t[2]};
    }
    tinygltf::Value::Object extras;
    extras["source"] = tinygltf::Value(source);
    extras["supported_as_khr_lights_punctual"] = tinygltf::Value(false);
    extras["reason"] = tinygltf::Value(reason);
    extras["raw"] = jsonToTinyValue(record);
    node.extras = tinygltf::Value(std::move(extras));
    scene.nodes.push_back((int)model.nodes.size());
    model.nodes.push_back(std::move(node));
}

static void addHssdPunctualLight(tinygltf::Model& model,
                                 tinygltf::Scene& scene,
                                 const json& record,
                                 const std::string& source,
                                 const std::string& type,
                                 double positive_intensity_scale,
                                 double negative_intensity_scale,
                                 HssdLightingImport& stats) {
    double position[3] = {0, 0, 0};
    double direction[3] = {0, 0, -1};
    bool has_position = false;
    bool has_direction = false;
    readLightVector(record, type, position, direction, has_position, has_direction);

    tinygltf::Light light;
    light.name = record.value("name", std::string("hssd_light_") + std::to_string(model.lights.size()));
    light.type = type;
    double color[3] = {1, 1, 1};
    if (readVec3ByKeys(record, {"color", "diffuse_color", "diffuseColor"}, color)) {
        light.color = {color[0], color[1], color[2]};
    } else {
        light.color = {1, 1, 1};
    }
    double intensity = 1.0;
    readNumberByKeys(record, {"intensity", "intensity_scale", "intensityScale"}, intensity);
    light.intensity = intensity * (intensity >= 0 ? positive_intensity_scale : negative_intensity_scale);
    readNumberByKeys(record, {"range", "attenuation_range", "attenuationRange"}, light.range);
    if (type == "spot") {
        if (record.contains("spot") && record["spot"].is_object()) {
            readNumberByKeys(record["spot"], {"innerConeAngle", "inner_cone_angle"}, light.spot.innerConeAngle);
            readNumberByKeys(record["spot"], {"outerConeAngle", "outer_cone_angle"}, light.spot.outerConeAngle);
        }
        readNumberByKeys(record, {"innerConeAngle", "inner_cone_angle"}, light.spot.innerConeAngle);
        readNumberByKeys(record, {"outerConeAngle", "outer_cone_angle"}, light.spot.outerConeAngle);
    }
    tinygltf::Value::Object lightExtras;
    lightExtras["hssd_source"] = tinygltf::Value(source);
    lightExtras["hssd_raw"] = jsonToTinyValue(record);
    light.extras = tinygltf::Value(std::move(lightExtras));

    const int lightIndex = (int)model.lights.size();
    model.lights.push_back(std::move(light));

    tinygltf::Node node;
    node.name = model.lights.back().name + "_node";
    node.light = lightIndex;
    if (record.contains("matrix") && record["matrix"].is_array() && record["matrix"].size() >= 16) {
        for (int i = 0; i < 16; i++) node.matrix.push_back(record["matrix"][i].get<double>());
    } else if ((type == "spot" || type == "directional") && has_direction) {
        const auto matrix = buildLightDirectionMatrix(position, direction);
        node.matrix.assign(matrix.begin(), matrix.end());
    } else if (record.contains("rotation") && record["rotation"].is_array() && record["rotation"].size() >= 4) {
        for (int i = 0; i < 4; i++) node.rotation.push_back(record["rotation"][i].get<double>());
        if (has_position) node.translation = {position[0], position[1], position[2]};
    } else if (has_position) {
        node.translation = {position[0], position[1], position[2]};
    }
    tinygltf::Value::Object nodeExtras;
    nodeExtras["hssd_source"] = tinygltf::Value(source);
    nodeExtras["hssd_raw"] = jsonToTinyValue(record);
    node.extras = tinygltf::Value(std::move(nodeExtras));

    scene.nodes.push_back((int)model.nodes.size());
    model.nodes.push_back(std::move(node));
    stats.punctual++;
}

static bool hasUnsupportedHssdPositionModel(const json& record,
                                            std::string& positionModel) {
    if (!readStringByKeys(record, {"position_model", "positionModel", "model"}, positionModel)) {
        return false;
    }
    positionModel = lowerString(positionModel);
    return positionModel != "global";
}

static HssdLightingImport importHssdLighting(tinygltf::Model& model,
                                             tinygltf::Scene& scene,
                                             const json& sceneJson,
                                             const fs::path& datasetRoot,
                                             const std::string& overridePath) {
    HssdLightingImport stats;
    tinygltf::Value::Object hssdLighting;

    if (sceneJson.contains("default_lighting")) {
        addRawHssdLightingExtras(hssdLighting, "default_lighting", sceneJson["default_lighting"]);
    }
    if (sceneJson.contains("default_pbr_shader_config")) {
        addRawHssdLightingExtras(hssdLighting, "default_pbr_shader_config", sceneJson["default_pbr_shader_config"]);
    }
    if (sceneJson.contains("pbr_shader_region_configs")) {
        addRawHssdLightingExtras(hssdLighting, "pbr_shader_region_configs", sceneJson["pbr_shader_region_configs"]);
    }

    std::vector<HssdLightingSource> sources;
    auto addHabitatDefaultLighting = [&](const std::string& source) {
        sources.push_back(habitatDefaultLightingSource());
        hssdLighting["default_lighting"] = tinygltf::Value("");
        hssdLighting["default_lighting_source"] = tinygltf::Value(source);
        hssdLighting["resolved_default_lighting"] = tinygltf::Value("habitat_default");
        stats.found_source = true;
    };
    auto addSourceFromPath = [&](const std::string& label, const std::string& pathText) {
        if (pathText.empty()) return;
        fs::path path = resolveHssdReferencePath(datasetRoot, pathText);
        json value;
        if (readJsonFile(path, value)) {
            HssdLightingSource source;
            source.label = label + ":" + path.string();
            source.value = std::move(value);
            readNumberByKeys(source.value, {"positive_intensity_scale", "positiveIntensityScale"},
                             source.positive_intensity_scale);
            readNumberByKeys(source.value, {"negative_intensity_scale", "negativeIntensityScale"},
                             source.negative_intensity_scale);
            sources.push_back(std::move(source));
            stats.found_source = true;
        } else {
            stats.missing_sources++;
            tinygltf::Value::Array missing;
            if (hssdLighting.count("missing_sources") && hssdLighting["missing_sources"].IsArray()) {
                missing = hssdLighting["missing_sources"].Get<tinygltf::Value::Array>();
            }
            missing.push_back(tinygltf::Value(path.string()));
            hssdLighting["missing_sources"] = tinygltf::Value(std::move(missing));
        }
    };

    if (!overridePath.empty()) {
        addSourceFromPath("override", overridePath);
        hssdLighting["override_lighting"] = tinygltf::Value(overridePath);
    } else if (sceneJson.contains("default_lighting")) {
        if (sceneJson["default_lighting"].is_string()) {
            const std::string lightSetupKey = sceneJson["default_lighting"].get<std::string>();
            const std::string lowerLightSetupKey = lowerString(lightSetupKey);
            if (lightSetupKey.empty()) {
                addHabitatDefaultLighting("scene_instance");
            } else if (lowerLightSetupKey == "no_lights") {
                hssdLighting["resolved_default_lighting"] = tinygltf::Value("no_lights");
                stats.found_source = true;
            } else {
                addSourceFromPath("default_lighting", lightSetupKey);
            }
        } else if (sceneJson["default_lighting"].is_object() || sceneJson["default_lighting"].is_array()) {
            HssdLightingSource source;
            source.label = "default_lighting:inline";
            source.value = sceneJson["default_lighting"];
            readNumberByKeys(source.value, {"positive_intensity_scale", "positiveIntensityScale"},
                             source.positive_intensity_scale);
            readNumberByKeys(source.value, {"negative_intensity_scale", "negativeIntensityScale"},
                             source.negative_intensity_scale);
            sources.push_back(std::move(source));
            stats.found_source = true;
        }
    } else {
        addHabitatDefaultLighting("scene_dataset_default");
    }

    for (const auto& source : sources) {
        std::vector<json> records;
        collectHssdLightRecords(source.value, records);
        if (records.empty() && isEnvironmentLightingRecord(source.value)) {
            records.push_back(source.value);
        }
        if (records.empty()) {
            tinygltf::Value::Array unparsed;
            if (hssdLighting.count("unparsed_sources") && hssdLighting["unparsed_sources"].IsArray()) {
                unparsed = hssdLighting["unparsed_sources"].Get<tinygltf::Value::Array>();
            }
            tinygltf::Value::Object item;
            item["source"] = tinygltf::Value(source.label);
            item["raw"] = jsonToTinyValue(source.value);
            unparsed.push_back(tinygltf::Value(std::move(item)));
            hssdLighting["unparsed_sources"] = tinygltf::Value(std::move(unparsed));
        }
        for (const auto& record : records) {
            const std::string type = lightTypeFromRecord(record);
            double intensity = 1.0;
            readNumberByKeys(record, {"intensity", "intensity_scale", "intensityScale"}, intensity);
            std::string positionModel;
            if (type == "point" || type == "spot" || type == "directional") {
                if (intensity < 0) {
                    stats.unsupported++;
                    addHssdUnsupportedLightNode(model, scene, record, source.label, "negative_intensity_light");
                } else if (hasUnsupportedHssdPositionModel(record, positionModel)) {
                    stats.unsupported++;
                    addHssdUnsupportedLightNode(model, scene, record, source.label,
                                                "non_global_position_model:" + positionModel);
                } else {
                    addHssdPunctualLight(model, scene, record, source.label, type,
                                         source.positive_intensity_scale,
                                         source.negative_intensity_scale,
                                         stats);
                }
            } else if (isEnvironmentLightingRecord(record)) {
                stats.environment++;
                addHssdUnsupportedLightNode(model, scene, record, source.label, "environment_or_ambient_light");
            } else {
                stats.unsupported++;
                addHssdUnsupportedLightNode(model, scene, record, source.label, "unsupported_light_type");
            }
        }
    }

    hssdLighting["punctual_lights_emitted"] = tinygltf::Value(stats.punctual);
    hssdLighting["unsupported_lights_preserved"] = tinygltf::Value(stats.unsupported);
    hssdLighting["environment_records_preserved"] = tinygltf::Value(stats.environment);
    hssdLighting["missing_source_count"] = tinygltf::Value(stats.missing_sources);

    tinygltf::Value::Object modelExtras;
    if (model.extras.IsObject()) {
        modelExtras = model.extras.Get<tinygltf::Value::Object>();
    }
    modelExtras["hssd_lighting"] = tinygltf::Value(std::move(hssdLighting));
    model.extras = tinygltf::Value(std::move(modelExtras));
    return stats;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: procthor2glb <scene_instance.json> [-o output.glb] [--normalize] [--hssd] [--hssd-lighting file.json]\n";
        return 1;
    }

    std::string inputPath;
    std::string outputPath;
    std::string hssdLightingPath;
    bool normalize = false;
    DatasetMode datasetMode = DatasetMode::AI2THOR;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        const std::string hssdLightingPrefix = "--hssd-lighting=";
        if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg == "--normalize") {
            normalize = true;
        } else if (arg == "--hssd") {
            datasetMode = DatasetMode::HSSD;
        } else if (arg == "--hssd-lighting") {
            if (i + 1 >= argc) {
                std::cerr << "error: missing value for --hssd-lighting\n";
                return 1;
            }
            hssdLightingPath = argv[++i];
        } else if (arg.compare(0, hssdLightingPrefix.size(), hssdLightingPrefix) == 0) {
            hssdLightingPath = arg.substr(hssdLightingPrefix.size());
        } else if (arg == "-h" || arg == "--help") {
            std::cout
                << "Usage: procthor2glb <scene_instance.json> [-o output.glb] "
                   "[--normalize] [--hssd] [--hssd-lighting file.json]\n"
                << "\nConvert a Habitat ai2thor-hab or HSSD scene to a "
                   "self-contained GLB.\n\n"
                << "Options:\n"
                << "  -o, --output <file>  Output GLB path (default: "
                   "<scene_name>.glb)\n"
                << "  --normalize          Decode KTX2 to PNG, dequantize meshes,\n"
                << "                       and bake texture transforms into UVs\n"
                << "                       for maximum viewer compatibility\n"
                << "  --hssd               Resolve stages/ and objects/ in an HSSD\n"
                << "                       dataset checkout\n"
                << "  --hssd-lighting <file>\n"
                << "                       Import an explicit HSSD/Habitat lighting\n"
                << "                       JSON config instead of scene default_lighting\n"
                << "  -h, --help           Show this help message\n";
            return 0;
        } else {
            inputPath = arg;
        }
    }

    if (inputPath.empty()) {
        std::cerr << "error: no input scene_instance.json specified\n";
        return 1;
    }

    // -- Read scene_instance.json --
    std::ifstream ifs(inputPath);
    if (!ifs.is_open()) {
        std::cerr << "error: cannot open " << inputPath << "\n";
        return 1;
    }
    json sceneJson;
    try {
        sceneJson = json::parse(ifs);
    } catch (const json::exception& e) {
        std::cerr << "error: failed to parse JSON: " << e.what() << "\n";
        return 1;
    }

    // -- Determine dataset root --
    fs::path datasetRoot = findDatasetRoot(inputPath, datasetMode);
    std::cout << "Dataset mode: "
              << (datasetMode == DatasetMode::HSSD ? "hssd" : "ai2thor-hab")
              << "\n";
    std::cout << "Dataset root: " << datasetRoot << "\n";

    // -- Default output name --
    if (outputPath.empty()) {
        std::string s = fs::path(inputPath).stem().string();
        auto pos = s.find(".scene_instance");
        if (pos != std::string::npos) s = s.substr(0, pos);
        outputPath = s + ".glb";
    }
    std::cout << "Output: " << outputPath << "\n";

    // -- Build output model --
    tinygltf::Model outModel;
    outModel.asset.version   = "2.0";
    outModel.asset.generator = "procthor2glb";

    tinygltf::Scene outScene;
    outScene.name = "scene";

    GltfCache cache;

    // -- Load & merge stage --
    if (sceneJson.count("stage_instance")) {
        auto& stageInst = sceneJson["stage_instance"];
        std::string tmpl = stageInst.value("template_name", "");
        if (!tmpl.empty()) {
            std::string path = resolveStageAssetPath(datasetMode, datasetRoot, tmpl);
            std::cout << "Loading stage: " << fs::path(path).filename().string()
                      << "\n";
            const auto* m = cache.load(path);
            if (m) {
                // Read the stage_config.json for front/up orientation
                std::string configPath =
                    resolveStageConfigPath(datasetMode, datasetRoot, tmpl);
                std::array<double, 16> stageMatrix{};
                bool hasStageMatrix = false;

                std::ifstream cfgFile(configPath);
                if (cfgFile.is_open()) {
                    try {
                        json cfg = json::parse(cfgFile);
                        double front[3] = {0, 0, -1};
                        double up[3]    = {0, 1, 0};
                        if (cfg.count("front") && cfg["front"].size() >= 3) {
                            for (int i = 0; i < 3; i++)
                                front[i] = cfg["front"][i].get<double>();
                        }
                        if (cfg.count("up") && cfg["up"].size() >= 3) {
                            for (int i = 0; i < 3; i++)
                                up[i] = cfg["up"][i].get<double>();
                        }
                        // Check if correction is needed (not already standard)
                        bool needsCorrection =
                            (front[0] != 0 || front[1] != 0 || front[2] != -1) ||
                            (up[0] != 0 || up[1] != 1 || up[2] != 0);
                        if (needsCorrection) {
                            stageMatrix = buildFrontUpCorrection(front, up);
                            hasStageMatrix = true;
                            std::cout << "  Applying front/up correction "
                                      << "(front=[" << front[0] << ","
                                      << front[1] << "," << front[2] << "]"
                                      << " up=[" << up[0] << "," << up[1]
                                      << "," << up[2] << "])\n";
                        }
                    } catch (const json::exception& e) {
                        std::cerr << "  warn: failed to parse stage config: "
                                  << e.what() << "\n";
                    }
                } else {
                    std::cerr << "  warn: stage config not found: "
                              << configPath << "\n";
                }

                auto r = mergeModel(outModel, *m,
                                    hasStageMatrix ? &stageMatrix : nullptr);
                outScene.nodes.push_back(r.rootNode);
                mergeExtensions(outModel, *m);
                std::cout << "  -> meshes=" << m->meshes.size()
                          << "  materials=" << m->materials.size()
                          << "  textures=" << m->textures.size() << "\n";
            } else {
                std::cerr << "error: failed to load stage " << path << "\n";
                return 1;
            }
        }
    }

    // -- Load & merge object instances --
    int objCount = 0, objSkipped = 0;
    if (sceneJson.count("object_instances")) {
        auto& objects = sceneJson["object_instances"];
        std::cout << "Processing " << objects.size()
                  << " object instances...\n";

        for (auto& objInst : objects) {
            std::string tmpl = objInst.value("template_name", "");
            if (tmpl.empty()) continue;

            std::string path = resolveObjectAssetPath(datasetMode, datasetRoot, tmpl);
            const auto* objModel = cache.load(path);
            if (!objModel) {
                objSkipped++;
                continue;
            }

            // Translation
            double t[3] = {0, 0, 0};
            if (objInst.count("translation")) {
                auto& tr = objInst["translation"];
                for (int i = 0; i < 3 && i < (int)tr.size(); i++)
                    t[i] = tr[i].get<double>();
            }

            // Quaternion [w, x, y, z]
            Quat q{1, 0, 0, 0};
            if (objInst.count("rotation")) {
                auto& rot = objInst["rotation"];
                if (rot.size() >= 4) {
                    q.w = rot[0].get<double>();
                    q.x = rot[1].get<double>();
                    q.y = rot[2].get<double>();
                    q.z = rot[3].get<double>();
                }
            }

            // Non-uniform scale
            double s[3] = {1, 1, 1};
            if (datasetMode == DatasetMode::HSSD) {
                std::ifstream cfgFile(resolveObjectConfigPath(datasetMode, datasetRoot, tmpl));
                if (cfgFile.is_open()) {
                    try {
                        json cfg = json::parse(cfgFile);
                        readScale(cfg, s);
                    } catch (const json::exception& e) {
                        std::cerr << "  warn: failed to parse object config for "
                                  << tmpl << ": " << e.what() << "\n";
                    }
                }
            }
            readScale(objInst, s);

            auto matrix = buildTRS(t, q, s);
            auto r = mergeModel(outModel, *objModel, &matrix);
            outScene.nodes.push_back(r.rootNode);
            mergeExtensions(outModel, *objModel);
            objCount++;
        }
    }

    std::cout << "Merged " << objCount << " objects";
    if (objSkipped > 0)
        std::cout << " (" << objSkipped << " failed to load)";
    std::cout << "\n";

    if (sceneJson.count("articulated_object_instances") &&
        sceneJson["articulated_object_instances"].is_array() &&
        !sceneJson["articulated_object_instances"].empty()) {
        std::cerr << "warn: skipping "
                  << sceneJson["articulated_object_instances"].size()
                  << " articulated_object_instances; URDF joint conversion is not implemented\n";
    }

    if (datasetMode == DatasetMode::HSSD) {
        HssdLightingImport lightingStats = importHssdLighting(
            outModel,
            outScene,
            sceneJson,
            datasetRoot,
            hssdLightingPath
        );
        std::cout << "HSSD lights: " << lightingStats.punctual
                  << " punctual emitted, " << lightingStats.unsupported
                  << " unsupported preserved, " << lightingStats.environment
                  << " environment records preserved";
        if (!lightingStats.found_source && hssdLightingPath.empty()) {
            std::cout << " (no default_lighting config referenced by scene)";
        }
        if (lightingStats.missing_sources > 0) {
            std::cout << " (" << lightingStats.missing_sources << " missing source"
                      << (lightingStats.missing_sources == 1 ? "" : "s") << ")";
        }
        std::cout << "\n";
    }

    // -- Finalize scene --
    outModel.scenes.push_back(outScene);
    outModel.defaultScene = 0;

    if (normalize) {
        // -- Dequantize integer mesh attributes → FLOAT --
        dequantizeMeshes(outModel);

        // -- Promote KHR_texture_basisu → standard texture source --
        promoteBasisTextures(outModel);

        // -- Decode KTX2/Basis textures to PNG --
        decodeKTX2Images(outModel);

        // -- Bake KHR_texture_transform into UV coords --
        bakeTextureTransforms(outModel);

        // -- Strip extensions that are now unnecessary --
        stripExtensions(outModel, {
            "KHR_mesh_quantization",
            "KHR_texture_basisu",
            "KHR_texture_transform",
        });
    }

    // -- Deduplicate light node names (assimp requires unique names) --
    {
        std::map<std::string, int> lightNodeNameCount;
        for (auto& node : outModel.nodes) {
            if (node.extensions.count("KHR_lights_punctual")) {
                auto& name = node.name;
                int count = lightNodeNameCount[name]++;
                if (count > 0) {
                    name += "_" + std::to_string(count);
                }
            }
        }
        for (int i = 0; i < (int)outModel.lights.size(); i++) {
            auto& light = outModel.lights[i];
            if (light.name.empty()) {
                light.name = "light_" + std::to_string(i);
            }
        }
    }

    // -- Consolidate buffers --
    consolidateBuffers(outModel);

    // -- Write GLB --
    tinygltf::TinyGLTF writer;
    writer.SetImageWriter(noopImageWriter, nullptr);
    bool ok = writer.WriteGltfSceneToFile(&outModel, outputPath,
                                          true,   // embedImages
                                          true,   // embedBuffers
                                          true,   // prettyPrint (N/A for binary)
                                          true);  // writeBinary
    if (!ok) {
        std::cerr << "error: failed to write " << outputPath << "\n";
        return 1;
    }

    // -- Summary --
    auto fileSize = fs::file_size(outputPath);
    std::cout << "\nWrote " << outputPath << " ("
              << std::fixed << std::setprecision(1)
              << (fileSize / (1024.0 * 1024.0)) << " MB)\n";
    std::cout << "  Nodes:     " << outModel.nodes.size() << "\n";
    std::cout << "  Meshes:    " << outModel.meshes.size() << "\n";
    std::cout << "  Materials: " << outModel.materials.size() << "\n";
    std::cout << "  Textures:  " << outModel.textures.size() << "\n";
    std::cout << "  Images:    " << outModel.images.size() << "\n";

    return 0;
}

// procthor2glb: Convert a Habitat ai2thor-hab scene_instance.json into a
// single self-contained GLB file.
//
// Usage:
//   procthor2glb <scene_instance.json> [-o output.glb] [--normalize]
//
// The tool reads the scene_instance.json, loads the stage GLB and every
// referenced object GLB, applies per-instance transforms (translation,
// rotation, scale), and writes one merged GLB.  KTX2/Basis textures are
// decoded to PNG so the output is a standard glTF 2.0 file.

#include <algorithm>
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

    // --- Nodes ---
    for (auto node : src.nodes) {
        if (node.mesh >= 0) node.mesh += meshOff;
        if (node.skin >= 0) node.skin = -1;  // drop skins for now
        for (auto& c : node.children) c += nodeOff;
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

// ---------------------------------------------------------------------------
// Resolve asset path:  <dataset_root>/assets/<template_name>.glb
// ---------------------------------------------------------------------------
static std::string resolveAssetPath(const fs::path& datasetRoot,
                                    const std::string& templateName) {
    return (datasetRoot / "assets" / (templateName + ".glb")).string();
}

// ---------------------------------------------------------------------------
// Find dataset root (directory containing both "assets" and "configs").
// ---------------------------------------------------------------------------
static fs::path findDatasetRoot(const fs::path& sceneJsonPath) {
    fs::path dir = fs::canonical(sceneJsonPath).parent_path();
    for (int i = 0; i < 10; i++) {
        if (fs::exists(dir / "assets") && fs::exists(dir / "configs"))
            return dir;
        if (!dir.has_parent_path() || dir == dir.parent_path()) break;
        dir = dir.parent_path();
    }
    return fs::canonical(sceneJsonPath).parent_path().parent_path().parent_path();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: procthor2glb <scene_instance.json> [-o output.glb] [--normalize]\n";
        return 1;
    }

    std::string inputPath;
    std::string outputPath;
    bool normalize = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg == "--normalize") {
            normalize = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout
                << "Usage: procthor2glb <scene_instance.json> [-o output.glb] "
                   "[--normalize]\n"
                << "\nConvert a Habitat ai2thor-hab scene to a self-contained "
                   "GLB.\n\n"
                << "Options:\n"
                << "  -o, --output <file>  Output GLB path (default: "
                   "<scene_name>.glb)\n"
                << "  --normalize          Decode KTX2 to PNG, dequantize meshes,\n"
                << "                       and bake texture transforms into UVs\n"
                << "                       for maximum viewer compatibility\n"
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
    fs::path datasetRoot = findDatasetRoot(inputPath);
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
            std::string path = resolveAssetPath(datasetRoot, tmpl);
            std::cout << "Loading stage: " << fs::path(path).filename().string()
                      << "\n";
            const auto* m = cache.load(path);
            if (m) {
                // Read the stage_config.json for front/up orientation
                std::string configPath =
                    (datasetRoot / "configs" / (tmpl + ".stage_config.json"))
                        .string();
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

            std::string path = resolveAssetPath(datasetRoot, tmpl);
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
            if (objInst.count("non_uniform_scale")) {
                auto& sc = objInst["non_uniform_scale"];
                for (int i = 0; i < 3 && i < (int)sc.size(); i++)
                    s[i] = sc[i].get<double>();
            }

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

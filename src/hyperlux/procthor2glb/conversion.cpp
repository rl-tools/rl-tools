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
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
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

static constexpr double AI2THOR_AMBIENT = 0.6;
// matches the median luminance of HSSD renders (no ceilings) to ProcTHOR's at AI2THOR_AMBIENT (High shading, 320 free-space views each)
static constexpr double HSSD_AMBIENT = 0.7;

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

// Column-major 4x4 product a * b
static std::array<double, 16> matMul(const std::array<double, 16>& a,
                                     const std::array<double, 16>& b) {
    std::array<double, 16> out{};
    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++)
            for (int k = 0; k < 4; k++)
                out[col * 4 + row] += a[k * 4 + row] * b[col * 4 + k];
    return out;
}

static std::array<double, 16> nodeLocalMatrix(const tinygltf::Node& node) {
    if (node.matrix.size() == 16) {
        std::array<double, 16> m{};
        std::copy(node.matrix.begin(), node.matrix.end(), m.begin());
        return m;
    }
    double t[3] = {0, 0, 0};
    Quat q{1, 0, 0, 0};
    double s[3] = {1, 1, 1};
    if (node.translation.size() == 3)
        for (int i = 0; i < 3; i++) t[i] = node.translation[i];
    if (node.rotation.size() == 4) {  // glTF stores [x, y, z, w]
        q.x = node.rotation[0];
        q.y = node.rotation[1];
        q.z = node.rotation[2];
        q.w = node.rotation[3];
    }
    if (node.scale.size() == 3)
        for (int i = 0; i < 3; i++) s[i] = node.scale[i];
    return buildTRS(t, q, s);
}

// ---------------------------------------------------------------------------
// Merge a source glTF model into a destination model.
// Returns the index of a wrapper node (with optional transform) that parents
// all of the source scene's root nodes.
// ---------------------------------------------------------------------------
struct MergeResult {
    int rootNode;
};

struct MergedNodes {
    int nodeOffset;
    std::vector<int> sourceRoots;  // src-relative indices of the source scene's root nodes
};

static MergedNodes mergeModelNodes(tinygltf::Model& dst,
                                   const tinygltf::Model& src) {
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
        if (node.light >= 0) {
            node.light += lightOff;
        }
        for (auto& c : node.children) {
            c += nodeOff;
        }
        // TinyGLTF serializes KHR_lights_punctual from node.light.
        node.extensions.erase("KHR_lights_punctual");
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

    return {nodeOff, srcRoots};
}

static MergeResult mergeModel(tinygltf::Model& dst,
                              const tinygltf::Model& src,
                              const std::array<double, 16>* instanceMatrix) {
    const MergedNodes merged = mergeModelNodes(dst, src);

    // Create wrapper node with optional transform
    tinygltf::Node wrapper;
    wrapper.name = "instance";
    for (int r : merged.sourceRoots)
        wrapper.children.push_back(r + merged.nodeOffset);
    if (instanceMatrix)
        wrapper.matrix.assign(instanceMatrix->begin(), instanceMatrix->end());

    int wrapperIdx = (int)dst.nodes.size();
    dst.nodes.push_back(wrapper);
    return {wrapperIdx};
}

static bool nodeHasDirectContent(const tinygltf::Model& src, int idx) {
    const tinygltf::Node& node = src.nodes[idx];
    return node.mesh >= 0 || node.light >= 0;
}

static bool subtreeHasContent(const tinygltf::Model& src, int idx) {
    if (nodeHasDirectContent(src, idx)) return true;
    for (int c : src.nodes[idx].children)
        if (subtreeHasContent(src, c)) return true;
    return false;
}

static void collectSplitRoots(tinygltf::Model& dst,
                              const tinygltf::Model& src,
                              int nodeOff,
                              int idx,
                              const std::array<double, 16>& parentMatrix,
                              const std::string& fallbackName,
                              std::vector<int>& outRoots) {
    if (!subtreeHasContent(src, idx)) return;
    const tinygltf::Node& node = src.nodes[idx];
    bool splitHere = nodeHasDirectContent(src, idx);
    if (!splitHere)
        for (int c : node.children)
            if (nodeHasDirectContent(src, c)) { splitHere = true; break; }
    std::array<double, 16> composed = matMul(parentMatrix, nodeLocalMatrix(node));
    if (splitHere) {
        tinygltf::Node& promoted = dst.nodes[idx + nodeOff];
        promoted.matrix.assign(composed.begin(), composed.end());
        promoted.translation.clear();
        promoted.rotation.clear();
        promoted.scale.clear();
        if (promoted.name.empty())
            promoted.name = fallbackName + "#part" + std::to_string(outRoots.size());
        outRoots.push_back(idx + nodeOff);
    } else {
        for (int c : node.children) {
            collectSplitRoots(dst, src, nodeOff, c, composed, fallbackName, outRoots);
        }
        dst.nodes[idx + nodeOff].children.clear();
    }
}

// Split-merge for the stage: consumers downstream create one instance per
// scene-root node, so instead of welding the whole source under a single
// wrapper, promote each minimal subtree that carries content (a mesh or light
// on the node itself or on a direct child) to its own scene root, baking the
// ancestor transforms into the promoted node. For a ProcTHOR stage this turns
// Level -> Room -> Wall/Floor/Ceiling into one root per wall/floor/ceiling,
// keeping their semantic names; a stage whose root carries the mesh directly
// degenerates to a single root, matching the previous welded behavior.
static std::vector<int> mergeModelSplit(tinygltf::Model& dst,
                                        const tinygltf::Model& src,
                                        const std::array<double, 16>* instanceMatrix,
                                        const std::string& fallbackName) {
    const MergedNodes merged = mergeModelNodes(dst, src);
    std::array<double, 16> base = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    if (instanceMatrix) base = *instanceMatrix;
    std::vector<int> roots;
    for (int r : merged.sourceRoots)
        collectSplitRoots(dst, src, merged.nodeOffset, r, base, fallbackName, roots);
    return roots;
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
                };
                redirect(mat.pbrMetallicRoughness.baseColorTexture);
                redirect(mat.pbrMetallicRoughness.metallicRoughnessTexture);
                redirect(mat.normalTexture);
                redirect(mat.occlusionTexture);
                redirect(mat.emissiveTexture);
            }

            bakedCount += (int)prims.size();
            continue;
        }

        bakedCount += (int)prims.size();
    }

    for (auto& mat : model.materials) {
        auto removeTT = [](auto& texInfo) {
            texInfo.extensions.erase("KHR_texture_transform");
        };
        removeTT(mat.pbrMetallicRoughness.baseColorTexture);
        removeTT(mat.pbrMetallicRoughness.metallicRoughnessTexture);
        removeTT(mat.normalTexture);
        removeTT(mat.occlusionTexture);
        removeTT(mat.emissiveTexture);
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
// Each bufferView's bytes are stored once at a 4-byte aligned offset: merged
// instances of one object repeat their geometry and textures, which make up
// 38% of the HSSD scene bytes and lie beyond gzip's 32 KB window.
// ---------------------------------------------------------------------------
static void consolidateBuffers(tinygltf::Model& model) {
    if (model.buffers.empty()) return;

    std::vector<unsigned char> merged;
    std::unordered_map<std::string_view, size_t> stored;
    size_t viewBytes = 0;
    for (auto& bv : model.bufferViews) {
        if (bv.buffer < 0 || bv.buffer >= (int)model.buffers.size()) continue;
        const std::string_view bytes(reinterpret_cast<const char*>(model.buffers[bv.buffer].data.data()) + bv.byteOffset, bv.byteLength);
        viewBytes += bytes.size();
        auto [entry, inserted] = stored.emplace(bytes, 0);
        if (inserted) {
            merged.resize(merged.size() + (4 - merged.size() % 4) % 4, 0);
            entry->second = merged.size();
            merged.insert(merged.end(), bytes.begin(), bytes.end());
        }
        bv.buffer = 0;
        bv.byteOffset = entry->second;
    }

    std::cout << "Consolidated " << model.buffers.size() << " buffers: "
              << merged.size() << " of " << viewBytes << " bufferView bytes stored\n";

    model.buffers.clear();
    tinygltf::Buffer buf;
    buf.data = std::move(merged);
    model.buffers.push_back(std::move(buf));
}

static void repairMaterialBindings(tinygltf::Model& model) {
    for (auto& material : model.materials) {
        for (auto& component : material.pbrMetallicRoughness.baseColorFactor) {
            component = std::clamp(component, 0.0, 1.0);
        }
    }
    std::map<std::pair<int, unsigned int>, int> variants;
    for (auto& mesh : model.meshes) {
        for (auto& primitive : mesh.primitives) {
            if (primitive.material < 0) {
                continue;
            }
            auto material = model.materials[primitive.material];
            unsigned int removed = 0;
            const auto removeMissingUV = [&](auto& texture, unsigned int slot) {
                if (texture.index < 0) {
                    return;
                }
                int texCoord = texture.texCoord;
                const auto transform = texture.extensions.find("KHR_texture_transform");
                if (transform != texture.extensions.end() && transform->second.Has("texCoord")) {
                    texCoord = transform->second.Get("texCoord").GetNumberAsInt();
                }
                if (primitive.attributes.count("TEXCOORD_" + std::to_string(texCoord)) == 0) {
                    texture = {};
                    removed |= slot;
                }
            };
            removeMissingUV(material.pbrMetallicRoughness.baseColorTexture, 1u);
            removeMissingUV(material.pbrMetallicRoughness.metallicRoughnessTexture, 2u);
            removeMissingUV(material.normalTexture, 4u);
            removeMissingUV(material.occlusionTexture, 8u);
            removeMissingUV(material.emissiveTexture, 16u);
            if (removed == 0) {
                continue;
            }
            const auto key = std::make_pair(primitive.material, removed);
            const auto existing = variants.find(key);
            if (existing != variants.end()) {
                primitive.material = existing->second;
            } else {
                primitive.material = static_cast<int>(model.materials.size());
                variants.emplace(key, primitive.material);
                model.materials.push_back(std::move(material));
            }
        }
    }
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
    if(value.is_number_integer() && value.get<double>() >= std::numeric_limits<int>::min() && value.get<double>() <= std::numeric_limits<int>::max()){
        return tinygltf::Value(value.get<int>());
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

static std::string sceneId(const fs::path& sceneJsonPath) {
    std::string name = sceneJsonPath.stem().string();
    const auto suffix = name.find(".scene_instance");
    return suffix == std::string::npos ? name : name.substr(0, suffix);
}

// RFC 4180: quoted fields may hold commas, doubled quotes and line breaks (HSSD's condensed table header does).
static std::vector<std::vector<std::string>> readCsv(const fs::path& path){
    std::ifstream stream(path, std::ios::binary);
    if(!stream){
        throw std::runtime_error("cannot read semantic input " + path.string());
    }
    const std::string text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> row;
    std::string field;
    bool quoted = false;
    for(size_t index = 0; index < text.size(); index++){
        const char c = text[index];
        if(quoted){
            if(c != '"'){
                field += c;
            }
            else if(index + 1 < text.size() && text[index + 1] == '"'){
                field += '"';
                index++;
            }
            else{
                quoted = false;
            }
        }
        else if(c == '"'){
            quoted = true;
        }
        else if(c == ','){
            row.push_back(std::move(field));
            field.clear();
        }
        else if(c == '\n' || c == '\r'){
            if(c == '\r' && index + 1 < text.size() && text[index + 1] == '\n'){
                index++;
            }
            row.push_back(std::move(field));
            field.clear();
            rows.push_back(std::move(row));
            row.clear();
        }
        else{
            field += c;
        }
    }
    if(quoted){
        throw std::runtime_error("unterminated quoted field in " + path.string());
    }
    if(!field.empty() || !row.empty()){
        row.push_back(std::move(field));
        rows.push_back(std::move(row));
    }
    return rows;
}

struct CsvTable {
    fs::path path;
    std::vector<std::string> header;
    std::map<std::string, std::vector<std::string>> rows;

    size_t column(const std::string& prefix) const {
        for(size_t index = 0; index < header.size(); index++){
            if(header[index].rfind(prefix, 0) == 0){
                return index;
            }
        }
        throw std::runtime_error("no column " + prefix + " in " + path.string());
    }

    // decomposed parts (<hash>_part_<n>) fall back to their base object's row
    const std::vector<std::string>* find(const std::string& hash) const {
        auto row = rows.find(hash);
        const auto part = hash.find("_part_");
        if(row == rows.end() && part != std::string::npos){
            row = rows.find(hash.substr(0, part));
        }
        return row == rows.end() ? nullptr : &row->second;
    }
};

static CsvTable readCsvTable(const fs::path& path, const std::string& keyColumnName){
    auto records = readCsv(path);
    if(records.empty()){
        throw std::runtime_error("empty semantic input " + path.string());
    }
    CsvTable table{path, std::move(records.front()), {}};
    const size_t keyColumn = table.column(keyColumnName);
    for(size_t index = 1; index < records.size(); index++){
        auto& record = records[index];
        if(record.size() != table.header.size()){
            throw std::runtime_error("row " + std::to_string(index) + " of " + path.string() + " has " + std::to_string(record.size()) + " fields, expected " + std::to_string(table.header.size()));
        }
        const std::string key = record[keyColumn];
        if(!table.rows.emplace(key, std::move(record)).second){
            throw std::runtime_error("duplicate key " + key + " in " + path.string());
        }
    }
    return table;
}

struct SemanticInputs {
    json classes;
    json objects = json::object();
    json categories = json::object();  // HSSD: per template, the non-empty objects.csv and condensed-table fields
};

static std::set<uint32_t> classIds(const json& classes){
    std::set<uint32_t> ids;
    for(const auto& value : classes.items()){
        if(!value.value().is_number_integer() || value.value().get<int64_t>() < 0 || value.value().get<int64_t>() >= 0xFFFFFFFFLL || !ids.insert(value.value().get<uint32_t>()).second){
            throw std::runtime_error("invalid or duplicate semantic class ID: " + value.key());
        }
    }
    return ids;
}

static json readJsonInput(const fs::path& path){
    std::ifstream stream(path);
    if(!stream){
        throw std::runtime_error("cannot read semantic input " + path.string());
    }
    return json::parse(stream);
}

static SemanticInputs readAi2thorSemanticInputs(const fs::path& root, const json& scene){
    SemanticInputs inputs;
    inputs.classes = readJsonInput(root / "configs/object_semantic_id_mapping.json");
    if(!inputs.classes.is_object() || !inputs.classes.contains("Undefined") || inputs.classes["Undefined"] != 0){
        throw std::runtime_error("semantic taxonomy must define Undefined as 0");
    }
    const auto ids = classIds(inputs.classes);
    for(const auto& instance : scene.value("object_instances", json::array())){
        const std::string name = instance.value("template_name", "");
        if(name.empty() || inputs.objects.contains(name)){
            continue;
        }
        const auto config = readJsonInput(resolveObjectConfigPath(DatasetMode::AI2THOR, root, name));
        const auto id = config.value("semantic_id", json(0));
        if(!id.is_number_integer() || id.get<int64_t>() < 0 || id.get<int64_t>() >= 0xFFFFFFFFLL || !ids.count(id.get<uint32_t>())){
            throw std::runtime_error("object semantic_id absent from taxonomy: " + name);
        }
        inputs.objects[name] = id;
    }
    return inputs;
}

// HSSD object configs' semantic_id below 25 follow a stale ID space, so classes come from the dataset tables:
// the condensed category, else objects.csv main_category, matched to a lexicon class by exact name, else 0 (unknown).
static SemanticInputs readHssdSemanticInputs(const fs::path& root, const json& scene){
    SemanticInputs inputs;
    inputs.classes = json::object();
    const auto lexicon = readJsonInput(root / "semantics/hssd-hab_semantic_lexicon.json");
    for(const auto& entry : lexicon.at("classes")){
        const auto name = entry.at("name").get<std::string>();
        if(name.empty() || inputs.classes.contains(name)){
            throw std::runtime_error("empty or duplicate semantic class name: " + name);
        }
        inputs.classes[name] = entry.at("id");
    }
    if(!inputs.classes.contains("unknown") || inputs.classes["unknown"] != 0){
        throw std::runtime_error("semantic taxonomy must define unknown as 0");
    }
    classIds(inputs.classes);
    const auto condensed = readCsvTable(root / "metadata/hssd_obj_semantics_condensed.csv", "Object Hash");
    const auto objects = readCsvTable(root / "semantics/objects.csv", "id");
    const size_t condensedCategory = condensed.column("Semantic Category:\nCONDENSED");
    const size_t primaryCategory = condensed.column("Semantic Category:\nThe primary");
    const size_t pickable = condensed.column("Is Pickable:");
    const size_t wnsynsetkey = objects.column("wnsynsetkey");
    const size_t mainCategory = objects.column("main_category");
    const size_t superCategory = objects.column("super_category");
    for(const auto& instance : scene.value("object_instances", json::array())){
        const std::string name = instance.value("template_name", "");
        if(name.empty() || inputs.objects.contains(name)){
            continue;
        }
        const std::string hash = fs::path(name).filename().string();
        const auto* condensedRow = condensed.find(hash);
        const auto* objectsRow = objects.find(hash);
        json id = 0;
        if(condensedRow != nullptr && inputs.classes.contains((*condensedRow)[condensedCategory])){
            id = inputs.classes[(*condensedRow)[condensedCategory]];
        }
        else if(objectsRow != nullptr && inputs.classes.contains((*objectsRow)[mainCategory])){
            id = inputs.classes[(*objectsRow)[mainCategory]];
        }
        inputs.objects[name] = id;
        json record = json::object();
        const auto copy = [&](const std::vector<std::string>* row, size_t column, const char* key){
            if(row != nullptr && !(*row)[column].empty()){
                record[key] = (*row)[column];
            }
        };
        copy(objectsRow, wnsynsetkey, "wnsynsetkey");
        copy(objectsRow, mainCategory, "main_category");
        copy(objectsRow, superCategory, "super_category");
        copy(condensedRow, condensedCategory, "condensed_category");
        copy(condensedRow, primaryCategory, "primary_category");
        if(condensedRow != nullptr && ((*condensedRow)[pickable] == "Yes" || (*condensedRow)[pickable] == "No")){
            record["pickable"] = (*condensedRow)[pickable] == "Yes";
        }
        inputs.categories[name] = std::move(record);
    }
    return inputs;
}

static SemanticInputs readSemanticInputs(DatasetMode mode, const fs::path& root, const json& scene){
    return mode == DatasetMode::HSSD ? readHssdSemanticInputs(root, scene) : readAi2thorSemanticInputs(root, scene);
}

static void setSemanticClass(tinygltf::Node& node, uint32_t id){
    auto extras = node.extras.IsObject() ? node.extras.Get<tinygltf::Value::Object>() : tinygltf::Value::Object{};
    auto rl = extras.count("rl_tools") && extras["rl_tools"].IsObject() ? extras["rl_tools"].Get<tinygltf::Value::Object>() : tinygltf::Value::Object{};
    rl["segmentation_class"] = jsonToTinyValue(json(id));
    extras["rl_tools"] = tinygltf::Value(rl);
    node.extras = tinygltf::Value(extras);
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
#include <rl_tools/hyperlux/datasets/procthor/conversion.h>
#include <conta/conta.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::hyperlux::datasets::procthor::conversion {
bool semantic_source(const Options& options, std::string& source, std::string& error){
    try{
        std::ifstream stream(options.scene_instance_path);
        const auto scene = json::parse(stream);
        const DatasetMode mode = options.dataset == Dataset::HSSD ? DatasetMode::HSSD : DatasetMode::AI2THOR;
        const auto root = findDatasetRoot(options.scene_instance_path, mode);
        const auto inputs = readSemanticInputs(mode, root, scene);
        std::string stage_hash;
        const std::string stage = scene.value("stage_instance", json::object()).value("template_name", "");
        if(!stage.empty() && !conta::detail::sha1_file(resolveStageAssetPath(mode, root, stage), stage_hash)){
            throw std::runtime_error("cannot hash stage semantic input: " + stage);
        }
        source = json{{"classes", inputs.classes}, {"objects", inputs.objects}, {"stage", stage_hash}}.dump();
        return true;
    }
    catch(const std::exception& exception){
        error = exception.what();
        return false;
    }
}

bool scene_instance_to_glb(const Options& options, std::string& error) {
    const std::string& inputPath = options.scene_instance_path;
    std::string outputPath = options.output_glb_path;
    const std::string& hssdLightingPath = options.hssd_lighting_path;
    const bool normalize = options.normalize;
    DatasetMode datasetMode = options.dataset == Dataset::HSSD ? DatasetMode::HSSD : DatasetMode::AI2THOR;

    // -- Read scene_instance.json --
    std::ifstream ifs(inputPath);
    if (!ifs.is_open()) {
        error = "cannot open " + inputPath;
        return false;
    }
    json sceneJson;
    try {
        sceneJson = json::parse(ifs);
    } catch (const json::exception& e) {
        error = std::string("failed to parse JSON: ") + e.what();
        return false;
    }

    // -- Determine dataset root --
    fs::path datasetRoot = findDatasetRoot(inputPath, datasetMode);
    SemanticInputs semantics;
    json regionAnnotations;
    try{
        semantics = readSemanticInputs(datasetMode, datasetRoot, sceneJson);
        if(datasetMode == DatasetMode::HSSD){
            regionAnnotations = readJsonInput(datasetRoot / "semantics/scenes" / (sceneId(inputPath) + ".semantic_config.json")).at("region_annotations");
        }
    }
    catch(const std::exception& exception){
        error = exception.what();
        return false;
    }
    std::cout << "Dataset mode: "
              << (datasetMode == DatasetMode::HSSD ? "hssd" : "ai2thor-hab")
              << "\n";
    std::cout << "Dataset root: " << datasetRoot << "\n";

    // -- Default output name --
    if (outputPath.empty()) {
        outputPath = sceneId(inputPath) + ".glb";
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

                auto stageRoots = mergeModelSplit(outModel, *m,
                                                  hasStageMatrix ? &stageMatrix : nullptr,
                                                  fs::path(tmpl).stem().string());
                for(int r : stageRoots){
                    auto& node = outModel.nodes[r];
                    std::string type;
                    if(node.extras.IsObject() && node.extras.Has("type") && node.extras.Get("type").IsString()){
                        type = node.extras.Get("type").Get<std::string>();
                    }
                    setSemanticClass(node, semantics.classes.value(type, 0u));
                    outScene.nodes.push_back(r);
                }
                mergeExtensions(outModel, *m);
                std::cout << "  -> parts=" << stageRoots.size()
                          << "  meshes=" << m->meshes.size()
                          << "  materials=" << m->materials.size()
                          << "  textures=" << m->textures.size() << "\n";
            } else {
                error = "failed to load stage " + path;
                return false;
            }
        }
    }

    // -- Load & merge object instances --
    int objCount = 0, objSkipped = 0;
    if (sceneJson.count("object_instances")) {
        auto& objects = sceneJson["object_instances"];
        std::cout << "Processing " << objects.size()
                  << " object instances...\n";

        for (size_t instanceIndex = 0; instanceIndex < objects.size(); instanceIndex++) {
            auto& objInst = objects[instanceIndex];
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
                json cfg;
                if (!readJsonFile(resolveObjectConfigPath(datasetMode, datasetRoot, tmpl), cfg)) {
                    error = "no readable object config for merged object " + tmpl;
                    return false;
                }
                readScale(cfg, s);
            }
            readScale(objInst, s);

            auto matrix = buildTRS(t, q, s);
            auto r = mergeModel(outModel, *objModel, &matrix);
            auto& root = outModel.nodes[r.rootNode];
            root.name = fs::path(tmpl).stem().string();
            setSemanticClass(root, semantics.objects.at(tmpl).get<uint32_t>());
            if (datasetMode == DatasetMode::HSSD) {
                json record = {{"template", tmpl}, {"instance", instanceIndex}};
                if (!semantics.categories.at(tmpl).empty()) {
                    record["categories"] = semantics.categories.at(tmpl);
                }
                auto extras = root.extras.Get<tinygltf::Value::Object>();
                extras["hssd"] = jsonToTinyValue(record);
                root.extras = tinygltf::Value(std::move(extras));
            }
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

    {
        const bool hssd = datasetMode == DatasetMode::HSSD;
        const double ambient = hssd ? HSSD_AMBIENT : AI2THOR_AMBIENT;
        json sceneExtras = {{"rl_tools", {{"environment", {
            {"mode", "solid"}, {"ambient", {ambient, ambient, ambient}}, {"background", {0.0, 0.0, 0.0}}
        }}, {"segmentation", {{"taxonomy", hssd ? "hssd-hab" : "ai2thor-hab"}, {"classes", semantics.classes}}}}}};
        if (hssd) {
            sceneExtras["hssd"] = {{"scene", sceneId(inputPath)}, {"region_annotations", std::move(regionAnnotations)}};
        }
        outScene.extras = jsonToTinyValue(sceneExtras);
    }

    // -- Finalize scene --
    outModel.scenes.push_back(outScene);
    outModel.defaultScene = 0;

    repairMaterialBindings(outModel);

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
            if (node.light >= 0) {
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
        error = "failed to write " + outputPath;
        return false;
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

    return true;
}
}
RL_TOOLS_NAMESPACE_WRAPPER_END

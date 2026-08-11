#ifndef TESTS_RENDERING_RAYTRACING_GOLDEN_IO_H
#define TESTS_RENDERING_RAYTRACING_GOLDEN_IO_H

// Golden file I/O shared by generators and comparison tests. The original per-camera depth and
// probe functions retain their native legacy format; overlay buffers use the versioned, little-
// endian multi-camera format below (fixed header, then camera-major row-major payload; the wire
// field order is num_cameras, height, width).

// RL_TOOLS_STB_PROVIDED arbitrates between this header and the renderer's
// operations_cpu_common.h so that whichever is included first provides the static stb
// implementation exactly once per TU (stb's implementation section has no include guard).
#ifndef RL_TOOLS_STB_PROVIDED
#define RL_TOOLS_STB_PROVIDED
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#endif

#include <rl_tools/rendering/raytracing/types.h>

#include "golden_layout.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace golden {
    enum class MultiCameraElementType: uint32_t{
        FLOAT32 = 1,
        UINT32 = 2
    };

    struct MultiCameraBinaryInfo{
        uint32_t version = 0;
        MultiCameraElementType element_type = MultiCameraElementType::FLOAT32;
        uint32_t num_cameras = 0;
        uint32_t height = 0;
        uint32_t width = 0;
        uint64_t element_count = 0;
    };

    static constexpr uint32_t MULTI_CAMERA_BINARY_VERSION = 1;
    static constexpr uint32_t SEGMENTATION_BACKGROUND_ID = 0xFFFFFFFFu;
    static constexpr int GRID_COLUMNS = 2;
    static constexpr int GRID_ROWS = 2;
    static constexpr int GRID_MAX_CAMERAS = GRID_COLUMNS * GRID_ROWS;

    namespace detail {
        static constexpr unsigned char MULTI_CAMERA_BINARY_MAGIC[8] = {'R', 'L', 'T', 'M', 'C', 'A', 'M', 0};
        static constexpr uint32_t MULTI_CAMERA_BINARY_HEADER_SIZE = 40;

        inline bool checked_element_count(int num_cameras, int width, int height, size_t& count){
            if(num_cameras <= 0 || width <= 0 || height <= 0){
                return false;
            }
            const size_t cameras = (size_t)num_cameras;
            const size_t image_width = (size_t)width;
            const size_t image_height = (size_t)height;
            if(image_width > std::numeric_limits<size_t>::max() / image_height){
                return false;
            }
            const size_t camera_pixels = image_width * image_height;
            if(cameras > std::numeric_limits<size_t>::max() / camera_pixels){
                return false;
            }
            count = cameras * camera_pixels;
            return true;
        }

        inline bool checked_camera_grid_dimensions(
            int num_cameras,
            int camera_width,
            int camera_height,
            size_t& camera_pixel_count,
            size_t& grid_pixel_count
        ){
            return num_cameras <= GRID_MAX_CAMERAS
                && camera_width <= std::numeric_limits<int>::max() / GRID_COLUMNS
                && camera_height <= std::numeric_limits<int>::max() / GRID_ROWS
                && checked_element_count(num_cameras, camera_width, camera_height, camera_pixel_count)
                && checked_element_count(1, camera_width * GRID_COLUMNS, camera_height * GRID_ROWS, grid_pixel_count);
        }

        inline bool write_u32_le(FILE* file, uint32_t value){
            const unsigned char bytes[4] = {
                (unsigned char)(value & 0xFFu),
                (unsigned char)((value >> 8) & 0xFFu),
                (unsigned char)((value >> 16) & 0xFFu),
                (unsigned char)((value >> 24) & 0xFFu)
            };
            return std::fwrite(bytes, sizeof(bytes), 1, file) == 1;
        }

        inline bool write_u64_le(FILE* file, uint64_t value){
            const unsigned char bytes[8] = {
                (unsigned char)(value & 0xFFu),
                (unsigned char)((value >> 8) & 0xFFu),
                (unsigned char)((value >> 16) & 0xFFu),
                (unsigned char)((value >> 24) & 0xFFu),
                (unsigned char)((value >> 32) & 0xFFu),
                (unsigned char)((value >> 40) & 0xFFu),
                (unsigned char)((value >> 48) & 0xFFu),
                (unsigned char)((value >> 56) & 0xFFu)
            };
            return std::fwrite(bytes, sizeof(bytes), 1, file) == 1;
        }

        inline bool read_u32_le(FILE* file, uint32_t& value){
            unsigned char bytes[4];
            if(std::fread(bytes, sizeof(bytes), 1, file) != 1){
                return false;
            }
            value = (uint32_t)bytes[0]
                  | ((uint32_t)bytes[1] << 8)
                  | ((uint32_t)bytes[2] << 16)
                  | ((uint32_t)bytes[3] << 24);
            return true;
        }

        inline bool read_u64_le(FILE* file, uint64_t& value){
            unsigned char bytes[8];
            if(std::fread(bytes, sizeof(bytes), 1, file) != 1){
                return false;
            }
            value = (uint64_t)bytes[0]
                  | ((uint64_t)bytes[1] << 8)
                  | ((uint64_t)bytes[2] << 16)
                  | ((uint64_t)bytes[3] << 24)
                  | ((uint64_t)bytes[4] << 32)
                  | ((uint64_t)bytes[5] << 40)
                  | ((uint64_t)bytes[6] << 48)
                  | ((uint64_t)bytes[7] << 56);
            return true;
        }

        inline bool write_multi_camera_header(
            FILE* file,
            MultiCameraElementType element_type,
            int num_cameras,
            int width,
            int height,
            size_t element_count
        ){
            return std::fwrite(MULTI_CAMERA_BINARY_MAGIC, sizeof(MULTI_CAMERA_BINARY_MAGIC), 1, file) == 1
                && write_u32_le(file, MULTI_CAMERA_BINARY_VERSION)
                && write_u32_le(file, MULTI_CAMERA_BINARY_HEADER_SIZE)
                && write_u32_le(file, (uint32_t)element_type)
                && write_u32_le(file, (uint32_t)num_cameras)
                && write_u32_le(file, (uint32_t)height)
                && write_u32_le(file, (uint32_t)width)
                && write_u64_le(file, (uint64_t)element_count);
        }

        inline bool read_multi_camera_header(FILE* file, MultiCameraBinaryInfo& info){
            unsigned char magic[sizeof(MULTI_CAMERA_BINARY_MAGIC)];
            uint32_t header_size = 0;
            uint32_t element_type = 0;
            if(std::fread(magic, sizeof(magic), 1, file) != 1
                || std::memcmp(magic, MULTI_CAMERA_BINARY_MAGIC, sizeof(magic)) != 0
                || !read_u32_le(file, info.version)
                || !read_u32_le(file, header_size)
                || !read_u32_le(file, element_type)
                || !read_u32_le(file, info.num_cameras)
                || !read_u32_le(file, info.height)
                || !read_u32_le(file, info.width)
                || !read_u64_le(file, info.element_count)){
                return false;
            }
            if(info.version != MULTI_CAMERA_BINARY_VERSION || header_size != MULTI_CAMERA_BINARY_HEADER_SIZE){
                return false;
            }
            if(element_type != (uint32_t)MultiCameraElementType::FLOAT32
                && element_type != (uint32_t)MultiCameraElementType::UINT32){
                return false;
            }
            info.element_type = (MultiCameraElementType)element_type;
            size_t expected_count = 0;
            return info.num_cameras <= (uint32_t)std::numeric_limits<int>::max()
                && info.width <= (uint32_t)std::numeric_limits<int>::max()
                && info.height <= (uint32_t)std::numeric_limits<int>::max()
                && checked_element_count((int)info.num_cameras, (int)info.width, (int)info.height, expected_count)
                && info.element_count == (uint64_t)expected_count;
        }

        inline bool has_no_trailing_data(FILE* file){
            unsigned char byte = 0;
            return std::fread(&byte, 1, 1, file) == 0 && std::feof(file) != 0;
        }

        inline bool header_matches(
            const MultiCameraBinaryInfo& info,
            MultiCameraElementType element_type,
            int expected_num_cameras,
            int expected_width,
            int expected_height
        ){
            return expected_num_cameras > 0 && expected_width > 0 && expected_height > 0
                && info.element_type == element_type
                && info.num_cameras == (uint32_t)expected_num_cameras
                && info.width == (uint32_t)expected_width
                && info.height == (uint32_t)expected_height;
        }

        inline bool write_multi_camera_bin(
            const std::string& path,
            MultiCameraElementType element_type,
            const uint32_t* bits,
            int num_cameras,
            int width,
            int height
        ){
            size_t count = 0;
            if(bits == nullptr || !checked_element_count(num_cameras, width, height, count)){
                return false;
            }
            FILE* file = std::fopen(path.c_str(), "wb");
            if(file == nullptr){
                return false;
            }
            bool ok = write_multi_camera_header(file, element_type, num_cameras, width, height, count);
            for(size_t value_i = 0; ok && value_i < count; value_i++){
                ok = write_u32_le(file, bits[value_i]);
            }
            ok = std::fclose(file) == 0 && ok;
            return ok;
        }

        inline bool load_multi_camera_bin(
            const std::string& path,
            MultiCameraElementType element_type,
            int expected_num_cameras,
            int expected_width,
            int expected_height,
            std::vector<uint32_t>& bits
        ){
            FILE* file = std::fopen(path.c_str(), "rb");
            if(file == nullptr){
                return false;
            }
            MultiCameraBinaryInfo info;
            bool ok = read_multi_camera_header(file, info)
                && header_matches(info, element_type, expected_num_cameras, expected_width, expected_height);
            std::vector<uint32_t> loaded;
            if(ok){
                loaded.resize((size_t)info.element_count);
                for(size_t value_i = 0; ok && value_i < loaded.size(); value_i++){
                    ok = read_u32_le(file, loaded[value_i]);
                }
                ok = ok && has_no_trailing_data(file);
            }
            ok = std::fclose(file) == 0 && ok;
            if(ok){
                bits.swap(loaded);
            }
            return ok;
        }
    }

    inline bool write_multi_camera_uint32_bin(
        const std::string& path,
        const uint32_t* values,
        int num_cameras,
        int width,
        int height
    ){
        return detail::write_multi_camera_bin(path, MultiCameraElementType::UINT32, values, num_cameras, width, height);
    }

    inline bool write_multi_camera_float_bin(
        const std::string& path,
        const float* values,
        int num_cameras,
        int width,
        int height
    ){
        static_assert(sizeof(float) == sizeof(uint32_t) && std::numeric_limits<float>::is_iec559,
                      "golden float binaries require IEEE-754 32-bit floats");
        size_t count = 0;
        if(values == nullptr || !detail::checked_element_count(num_cameras, width, height, count)){
            return false;
        }
        std::vector<uint32_t> bits(count);
        std::memcpy(bits.data(), values, count * sizeof(uint32_t));
        return detail::write_multi_camera_bin(path, MultiCameraElementType::FLOAT32, bits.data(), num_cameras, width, height);
    }

    inline bool load_multi_camera_uint32_bin(
        const std::string& path,
        int expected_num_cameras,
        int expected_width,
        int expected_height,
        std::vector<uint32_t>& values
    ){
        return detail::load_multi_camera_bin(path, MultiCameraElementType::UINT32, expected_num_cameras, expected_width, expected_height, values);
    }

    inline bool load_multi_camera_float_bin(
        const std::string& path,
        int expected_num_cameras,
        int expected_width,
        int expected_height,
        std::vector<float>& values
    ){
        static_assert(sizeof(float) == sizeof(uint32_t) && std::numeric_limits<float>::is_iec559,
                      "golden float binaries require IEEE-754 32-bit floats");
        std::vector<uint32_t> bits;
        if(!detail::load_multi_camera_bin(path, MultiCameraElementType::FLOAT32, expected_num_cameras, expected_width, expected_height, bits)){
            return false;
        }
        std::vector<float> loaded(bits.size());
        std::memcpy(loaded.data(), bits.data(), bits.size() * sizeof(uint32_t));
        values.swap(loaded);
        return true;
    }

    inline bool write_camera_png(const std::string& path, const uint32_t* pixels, int width, int height){
        return stbi_write_png(path.c_str(), width, height, 4, pixels, width * 4) != 0;
    }

    inline bool load_camera_png(const std::string& path, int expected_width, int expected_height, std::vector<uint32_t>& pixels){
        int width, height, channels;
        unsigned char* image = stbi_load(path.c_str(), &width, &height, &channels, 4);
        if(image == nullptr){
            return false;
        }
        const bool ok = width == expected_width && height == expected_height;
        if(ok){
            pixels.resize((size_t)width * height);
            std::memcpy(pixels.data(), image, (size_t)width * height * 4);
        }
        stbi_image_free(image);
        return ok;
    }

    inline bool make_camera_grid(
        const uint32_t* camera_major_pixels,
        int num_cameras,
        int camera_width,
        int camera_height,
        std::vector<uint32_t>& grid_pixels
    ){
        size_t camera_pixel_count = 0;
        size_t grid_pixel_count = 0;
        if(camera_major_pixels == nullptr
            || !detail::checked_camera_grid_dimensions(
                num_cameras,
                camera_width,
                camera_height,
                camera_pixel_count,
                grid_pixel_count
            )){
            return false;
        }
        std::vector<uint32_t> grid(grid_pixel_count, 0xFF000000u);
        const int grid_width = camera_width * GRID_COLUMNS;
        const size_t pixels_per_camera = (size_t)camera_width * camera_height;
        for(int camera_i = 0; camera_i < num_cameras; camera_i++){
            const int offset_x = (camera_i % GRID_COLUMNS) * camera_width;
            const int offset_y = (camera_i / GRID_COLUMNS) * camera_height;
            for(int y = 0; y < camera_height; y++){
                std::memcpy(
                    grid.data() + (size_t)(offset_y + y) * grid_width + offset_x,
                    camera_major_pixels + (size_t)camera_i * pixels_per_camera + (size_t)y * camera_width,
                    (size_t)camera_width * sizeof(uint32_t)
                );
            }
        }
        grid_pixels.swap(grid);
        return true;
    }

    inline bool split_camera_grid(
        const uint32_t* grid_pixels,
        int num_cameras,
        int camera_width,
        int camera_height,
        std::vector<uint32_t>& camera_major_pixels
    ){
        size_t camera_pixel_count = 0;
        size_t grid_pixel_count = 0;
        if(grid_pixels == nullptr
            || !detail::checked_camera_grid_dimensions(
                num_cameras,
                camera_width,
                camera_height,
                camera_pixel_count,
                grid_pixel_count
            )){
            return false;
        }
        std::vector<uint32_t> cameras(camera_pixel_count);
        const int grid_width = camera_width * GRID_COLUMNS;
        const size_t pixels_per_camera = (size_t)camera_width * camera_height;
        for(int camera_i = 0; camera_i < num_cameras; camera_i++){
            const int offset_x = (camera_i % GRID_COLUMNS) * camera_width;
            const int offset_y = (camera_i / GRID_COLUMNS) * camera_height;
            for(int y = 0; y < camera_height; y++){
                std::memcpy(
                    cameras.data() + (size_t)camera_i * pixels_per_camera + (size_t)y * camera_width,
                    grid_pixels + (size_t)(offset_y + y) * grid_width + offset_x,
                    (size_t)camera_width * sizeof(uint32_t)
                );
            }
        }
        camera_major_pixels.swap(cameras);
        return true;
    }

    inline bool write_camera_grid_png(
        const std::string& path,
        const uint32_t* camera_major_pixels,
        int num_cameras,
        int camera_width,
        int camera_height
    ){
        std::vector<uint32_t> grid;
        return make_camera_grid(camera_major_pixels, num_cameras, camera_width, camera_height, grid)
            && write_camera_png(path, grid.data(), camera_width * GRID_COLUMNS, camera_height * GRID_ROWS);
    }

    inline bool load_camera_grid_png(
        const std::string& path,
        int expected_num_cameras,
        int expected_camera_width,
        int expected_camera_height,
        std::vector<uint32_t>& camera_major_pixels
    ){
        if(expected_num_cameras <= 0 || expected_num_cameras > GRID_MAX_CAMERAS
            || expected_camera_width <= 0 || expected_camera_height <= 0
            || expected_camera_width > std::numeric_limits<int>::max() / GRID_COLUMNS
            || expected_camera_height > std::numeric_limits<int>::max() / GRID_ROWS){
            return false;
        }
        std::vector<uint32_t> grid;
        return load_camera_png(path, expected_camera_width * GRID_COLUMNS, expected_camera_height * GRID_ROWS, grid)
            && split_camera_grid(grid.data(), expected_num_cameras, expected_camera_width, expected_camera_height, camera_major_pixels);
    }

    inline uint32_t rgba(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha = 255){
        return (uint32_t)red
             | ((uint32_t)green << 8)
             | ((uint32_t)blue << 16)
             | ((uint32_t)alpha << 24);
    }

    // per-channel abs diff, amplified so small deviations are visible (value = min(255, 8 * |a - b|))
    inline void colorize_rgb_diff(const uint32_t* target, const uint32_t* current, size_t count, uint32_t* image){
        const auto* target_bytes = (const unsigned char*)target;
        const auto* current_bytes = (const unsigned char*)current;
        auto* image_bytes = (unsigned char*)image;
        for(size_t pixel_i = 0; pixel_i < count; pixel_i++){
            for(int channel = 0; channel < 3; channel++){
                const int delta = (int)current_bytes[pixel_i * 4 + channel] - (int)target_bytes[pixel_i * 4 + channel];
                const int amplified = (delta < 0 ? -delta : delta) * 8;
                image_bytes[pixel_i * 4 + channel] = (unsigned char)std::min(amplified, 255);
            }
            image_bytes[pixel_i * 4 + 3] = 255;
        }
    }

    inline bool write_camera_diff_png(const std::string& path, const uint32_t* ours, const uint32_t* golden, int width, int height){
        size_t count = 0;
        if(ours == nullptr || golden == nullptr || !detail::checked_element_count(1, width, height, count)){
            return false;
        }
        std::vector<uint32_t> image(count);
        colorize_rgb_diff(golden, ours, count, image.data());
        return write_camera_png(path, image.data(), width, height);
    }

    inline bool write_camera_grid_diff_png(
        const std::string& path,
        const uint32_t* current,
        const uint32_t* target,
        int num_cameras,
        int camera_width,
        int camera_height
    ){
        size_t count = 0;
        if(current == nullptr || target == nullptr
            || !detail::checked_element_count(num_cameras, camera_width, camera_height, count)){
            return false;
        }
        std::vector<uint32_t> diff(count);
        colorize_rgb_diff(target, current, count, diff.data());
        return write_camera_grid_png(path, diff.data(), num_cameras, camera_width, camera_height);
    }

    inline bool write_rgb_review_grid_pngs(
        const std::string& target_path,
        const std::string& current_path,
        const std::string& diff_path,
        const uint32_t* target,
        const uint32_t* current,
        int num_cameras,
        int camera_width,
        int camera_height
    ){
        return write_camera_grid_png(target_path, target, num_cameras, camera_width, camera_height)
            && write_camera_grid_png(current_path, current, num_cameras, camera_width, camera_height)
            && write_camera_grid_diff_png(diff_path, current, target, num_cameras, camera_width, camera_height);
    }

    // pinned normals encoding: round((clamp(n, -1, 1) * 0.5 + 0.5) * 255) per channel — must
    // match rendering::raytracing::detail::normal_to_rgba (operations_cpu_common.h), which owns
    // the renderer-side save verbs. Encoded normals reuse the rgb grid/diff/review machinery.
    inline uint32_t normal_rgba(const float normal[3]){
        uint8_t channels[3];
        for(int component = 0; component < 3; component++){
            const float clamped = std::min(std::max(normal[component], -1.f), 1.f);
            channels[component] = (uint8_t)std::lround((clamped * 0.5f + 0.5f) * 255.f);
        }
        return rgba(channels[0], channels[1], channels[2]);
    }

    inline void colorize_normals(const float* normals, size_t count, uint32_t* image){
        for(size_t pixel_i = 0; pixel_i < count; pixel_i++){
            image[pixel_i] = normal_rgba(&normals[pixel_i * 3]);
        }
    }

    // corpus surface for both the overlay and procthor_static_scene suites: segmentation.png
    // stores ids through this encoding and is validated against segmentation.bin, so the mapping
    // must never drift (pinned by GoldenIoTest.SegmentationEncodingIsPinned); every non-background
    // channel is >= 32, so the background gray (16,16,16) is unreachable for real ids
    inline uint32_t segmentation_false_color(uint32_t instance_id){
        if(instance_id == SEGMENTATION_BACKGROUND_ID){
            return rgba(16, 16, 16);
        }
        uint32_t hash = instance_id + 0x9E3779B9u;
        hash ^= hash >> 16;
        hash *= 0x7FEB352Du;
        hash ^= hash >> 15;
        hash *= 0x846CA68Bu;
        hash ^= hash >> 16;

        uint8_t channels[3] = {
            (uint8_t)(64u + (hash & 0x7Fu)),
            (uint8_t)(64u + ((hash >> 8) & 0x7Fu)),
            (uint8_t)(64u + ((hash >> 16) & 0x7Fu))
        };
        const int high_channel = (int)((hash >> 24) % 3u);
        const int low_channel = (high_channel + 1 + (int)((hash >> 30) & 1u)) % 3;
        channels[high_channel] = (uint8_t)(224u + ((hash >> 19) & 0x1Fu));
        channels[low_channel] = (uint8_t)(32u + ((hash >> 3) & 0x1Fu));
        return rgba(channels[0], channels[1], channels[2]);
    }

    inline uint32_t segmentation_diff_color(uint32_t target, uint32_t current){
        if(target == current){
            return rgba(0, 0, 0);
        }
        if(target == SEGMENTATION_BACKGROUND_ID){
            return rgba(40, 220, 100);
        }
        if(current == SEGMENTATION_BACKGROUND_ID){
            return rgba(245, 65, 55);
        }
        return rgba(235, 75, 235);
    }

    inline void colorize_segmentation(const uint32_t* segmentation, size_t count, uint32_t* image){
        for(size_t pixel_i = 0; pixel_i < count; pixel_i++){
            image[pixel_i] = segmentation_false_color(segmentation[pixel_i]);
        }
    }

    inline bool write_segmentation_grid_png(
        const std::string& path,
        const uint32_t* segmentation,
        int num_cameras,
        int camera_width,
        int camera_height
    ){
        size_t count = 0;
        if(segmentation == nullptr
            || !detail::checked_element_count(num_cameras, camera_width, camera_height, count)){
            return false;
        }
        std::vector<uint32_t> image(count);
        colorize_segmentation(segmentation, count, image.data());
        return write_camera_grid_png(path, image.data(), num_cameras, camera_width, camera_height);
    }

    inline bool write_segmentation_diff_grid_png(
        const std::string& path,
        const uint32_t* target,
        const uint32_t* current,
        int num_cameras,
        int camera_width,
        int camera_height
    ){
        size_t count = 0;
        if(target == nullptr || current == nullptr
            || !detail::checked_element_count(num_cameras, camera_width, camera_height, count)){
            return false;
        }
        std::vector<uint32_t> image(count);
        for(size_t pixel_i = 0; pixel_i < count; pixel_i++){
            image[pixel_i] = segmentation_diff_color(target[pixel_i], current[pixel_i]);
        }
        return write_camera_grid_png(path, image.data(), num_cameras, camera_width, camera_height);
    }

    inline bool write_segmentation_review_grid_pngs(
        const std::string& target_path,
        const std::string& current_path,
        const std::string& diff_path,
        const uint32_t* target,
        const uint32_t* current,
        int num_cameras,
        int camera_width,
        int camera_height
    ){
        return write_segmentation_grid_png(target_path, target, num_cameras, camera_width, camera_height)
            && write_segmentation_grid_png(current_path, current, num_cameras, camera_width, camera_height)
            && write_segmentation_diff_grid_png(diff_path, target, current, num_cameras, camera_width, camera_height);
    }

    struct DepthVisualizationRange{
        float minimum = 0;
        float maximum = 1;
        float valid_maximum = std::numeric_limits<float>::max();
        bool has_valid_values = false;
    };

    inline float depth_valid_maximum(float max_depth){
        return max_depth > 0.f && max_depth < std::numeric_limits<float>::max()
            ? max_depth * 0.999f
            : std::numeric_limits<float>::max();
    }

    inline bool valid_visualization_depth(float value, const DepthVisualizationRange& range){
        return value > 0.f && value < range.valid_maximum;
    }

    inline DepthVisualizationRange depth_visualization_range(
        const float* first,
        const float* second,
        size_t count,
        float max_depth
    ){
        DepthVisualizationRange range;
        range.valid_maximum = depth_valid_maximum(max_depth);
        if(first == nullptr){
            return range;
        }
        float minimum = std::numeric_limits<float>::max();
        float maximum = std::numeric_limits<float>::lowest();
        for(size_t value_i = 0; value_i < count; value_i++){
            const float first_value = first[value_i];
            if(first_value > 0.f && first_value < range.valid_maximum){
                minimum = std::min(minimum, first_value);
                maximum = std::max(maximum, first_value);
            }
            if(second != nullptr){
                const float second_value = second[value_i];
                if(second_value > 0.f && second_value < range.valid_maximum){
                    minimum = std::min(minimum, second_value);
                    maximum = std::max(maximum, second_value);
                }
            }
        }
        range.has_valid_values = minimum <= maximum;
        if(range.has_valid_values){
            range.minimum = minimum;
            range.maximum = maximum;
        }
        return range;
    }

    inline DepthVisualizationRange depth_visualization_range(const float* values, size_t count, float max_depth){
        return depth_visualization_range(values, nullptr, count, max_depth);
    }

    inline float depth_visualization_span(const DepthVisualizationRange& range){
        if(!range.has_valid_values){
            return 1.f;
        }
        const float span = range.maximum - range.minimum;
        const float magnitude = std::max(std::max(range.maximum, -range.minimum), 1.f);
        return std::max(span, magnitude * 1e-6f);
    }

    inline void colorize_depth(
        const float* depth,
        size_t count,
        const DepthVisualizationRange& range,
        uint32_t* image
    ){
        const float raw_span = range.maximum - range.minimum;
        const float span = depth_visualization_span(range);
        for(size_t pixel_i = 0; pixel_i < count; pixel_i++){
            uint8_t gray = 0;
            if(range.has_valid_values && valid_visualization_depth(depth[pixel_i], range)){
                if(raw_span <= 0.f){
                    gray = 128;
                }
                else{
                    const float normalized = std::max(0.f, std::min(1.f, (depth[pixel_i] - range.minimum) / span));
                    gray = (uint8_t)(normalized * 255.f);
                }
            }
            image[pixel_i] = rgba(gray, gray, gray);
        }
    }

    inline void colorize_depth_diff(
        const float* target,
        const float* current,
        size_t count,
        const DepthVisualizationRange& range,
        uint32_t* image,
        float amplification = 8.f
    ){
        const float span = depth_visualization_span(range);
        const float valid_amplification = amplification > 0.f ? amplification : 1.f;
        for(size_t pixel_i = 0; pixel_i < count; pixel_i++){
            const bool target_valid = range.has_valid_values && valid_visualization_depth(target[pixel_i], range);
            const bool current_valid = range.has_valid_values && valid_visualization_depth(current[pixel_i], range);
            if(target_valid != current_valid){
                image[pixel_i] = rgba(255, 0, 255);
            }
            else if(!target_valid){
                image[pixel_i] = rgba(0, 0, 0);
            }
            else{
                const float delta = target[pixel_i] - current[pixel_i];
                const float absolute_delta = delta < 0.f ? -delta : delta;
                const float normalized = std::max(0.f, std::min(1.f, absolute_delta / span * valid_amplification));
                const uint8_t intensity = (uint8_t)(normalized * 255.f);
                image[pixel_i] = rgba(intensity, (uint8_t)(intensity / 4), 0);
            }
        }
    }

    inline bool write_camera_depth_png(const std::string& path, const float* depth, int width, int height, float max_depth){
        size_t count = 0;
        if(depth == nullptr || !detail::checked_element_count(1, width, height, count)){
            return false;
        }
        std::vector<uint32_t> image(count);
        colorize_depth(depth, count, depth_visualization_range(depth, count, max_depth), image.data());
        return write_camera_png(path, image.data(), width, height);
    }

    inline bool write_camera_depth_diff_png(const std::string& path, const float* target, const float* current, int width, int height, float max_depth){
        size_t count = 0;
        if(target == nullptr || current == nullptr || !detail::checked_element_count(1, width, height, count)){
            return false;
        }
        std::vector<uint32_t> image(count);
        colorize_depth_diff(target, current, count, depth_visualization_range(target, current, count, max_depth), image.data());
        return write_camera_png(path, image.data(), width, height);
    }

    inline bool write_depth_grid_png(
        const std::string& path,
        const float* depth,
        int num_cameras,
        int camera_width,
        int camera_height,
        const DepthVisualizationRange& range
    ){
        size_t count = 0;
        if(depth == nullptr
            || !detail::checked_element_count(num_cameras, camera_width, camera_height, count)){
            return false;
        }
        std::vector<uint32_t> image(count);
        colorize_depth(depth, count, range, image.data());
        return write_camera_grid_png(path, image.data(), num_cameras, camera_width, camera_height);
    }

    inline bool write_depth_grid_png(
        const std::string& path,
        const float* depth,
        int num_cameras,
        int camera_width,
        int camera_height,
        float max_depth
    ){
        size_t count = 0;
        if(depth == nullptr || !detail::checked_element_count(num_cameras, camera_width, camera_height, count)){
            return false;
        }
        return write_depth_grid_png(
            path,
            depth,
            num_cameras,
            camera_width,
            camera_height,
            depth_visualization_range(depth, count, max_depth)
        );
    }

    inline bool write_depth_diff_grid_png(
        const std::string& path,
        const float* target,
        const float* current,
        int num_cameras,
        int camera_width,
        int camera_height,
        const DepthVisualizationRange& range,
        float amplification = 8.f
    ){
        size_t count = 0;
        if(target == nullptr || current == nullptr
            || !detail::checked_element_count(num_cameras, camera_width, camera_height, count)){
            return false;
        }
        std::vector<uint32_t> image(count);
        colorize_depth_diff(target, current, count, range, image.data(), amplification);
        return write_camera_grid_png(path, image.data(), num_cameras, camera_width, camera_height);
    }

    inline bool write_depth_review_grid_pngs(
        const std::string& target_path,
        const std::string& current_path,
        const std::string& diff_path,
        const float* target,
        const float* current,
        int num_cameras,
        int camera_width,
        int camera_height,
        float max_depth,
        float diff_amplification = 8.f
    ){
        size_t count = 0;
        if(target == nullptr || current == nullptr
            || !detail::checked_element_count(num_cameras, camera_width, camera_height, count)){
            return false;
        }
        const DepthVisualizationRange range = depth_visualization_range(target, current, count, max_depth);
        return write_depth_grid_png(target_path, target, num_cameras, camera_width, camera_height, range)
            && write_depth_grid_png(current_path, current, num_cameras, camera_width, camera_height, range)
            && write_depth_diff_grid_png(diff_path, target, current, num_cameras, camera_width, camera_height, range, diff_amplification);
    }

    inline bool write_camera_depth_bin(const std::string& path, const float* depth, int width, int height){
        FILE* file = std::fopen(path.c_str(), "wb");
        if(file == nullptr){
            return false;
        }
        const int num_cameras = 1;
        bool ok = std::fwrite(&num_cameras, sizeof(int), 1, file) == 1
               && std::fwrite(&height, sizeof(int), 1, file) == 1
               && std::fwrite(&width, sizeof(int), 1, file) == 1
               && std::fwrite(depth, sizeof(float), (size_t)width * height, file) == (size_t)width * height;
        std::fclose(file);
        return ok;
    }

    inline bool load_camera_depth_bin(const std::string& path, int expected_width, int expected_height, std::vector<float>& depth){
        FILE* file = std::fopen(path.c_str(), "rb");
        if(file == nullptr){
            return false;
        }
        int num_cameras = 0, height = 0, width = 0;
        bool ok = std::fread(&num_cameras, sizeof(int), 1, file) == 1
               && std::fread(&height, sizeof(int), 1, file) == 1
               && std::fread(&width, sizeof(int), 1, file) == 1
               && num_cameras == 1 && height == expected_height && width == expected_width;
        if(ok){
            depth.resize((size_t)width * height);
            ok = std::fread(depth.data(), sizeof(float), depth.size(), file) == depth.size();
        }
        std::fclose(file);
        return ok;
    }

    inline bool write_camera_probes(const std::string& path, const rl_tools::rendering::raytracing::CollisionResult* probes, int num_probes){
        FILE* file = std::fopen(path.c_str(), "wb");
        if(file == nullptr){
            return false;
        }
        const int num_cameras = 1;
        bool ok = std::fwrite(&num_cameras, sizeof(int), 1, file) == 1
               && std::fwrite(&num_probes, sizeof(int), 1, file) == 1
               && std::fwrite(probes, sizeof(rl_tools::rendering::raytracing::CollisionResult), (size_t)num_probes, file) == (size_t)num_probes;
        std::fclose(file);
        return ok;
    }

    inline bool load_camera_probes(const std::string& path, int expected_probes, std::vector<rl_tools::rendering::raytracing::CollisionResult>& probes){
        FILE* file = std::fopen(path.c_str(), "rb");
        if(file == nullptr){
            return false;
        }
        int num_cameras = 0, num_probes = 0;
        bool ok = std::fread(&num_cameras, sizeof(int), 1, file) == 1
               && std::fread(&num_probes, sizeof(int), 1, file) == 1
               && num_cameras == 1 && num_probes == expected_probes;
        if(ok){
            probes.resize((size_t)num_probes);
            ok = std::fread(probes.data(), sizeof(rl_tools::rendering::raytracing::CollisionResult), probes.size(), file) == probes.size();
        }
        std::fclose(file);
        return ok;
    }
}

#endif

#ifndef TESTS_RENDERING_RAYTRACING_GOLDEN_IO_H
#define TESTS_RENDERING_RAYTRACING_GOLDEN_IO_H

// Per-camera golden file I/O, shared between the generator (generate_golden.cpp) and the
// comparison tests (golden_comparison.cpp). Layout: <golden_dir>/<pose_id>/<case>.png,
// <case>_depth.bin ([1, height, width] int header + float32), probes.bin ([1, num_probes] int
// header + CollisionResult). The including TU must provide the stb_image/stb_image_write
// declarations (they come with any raytracing backend include).
#include <rl_tools/rendering/raytracing/types.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace golden {
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

    // normalized grayscale visualization (min/max of the valid range per image)
    inline bool write_camera_depth_png(const std::string& path, const float* depth, int width, int height, float max_depth){
        const float valid_max_depth = max_depth * 0.999f;
        float min_valid = 3.4e38f, max_valid = -3.4e38f;
        for(int pixel_i = 0; pixel_i < width * height; pixel_i++){
            const float value = depth[pixel_i];
            if(value > 0.f && value < valid_max_depth){
                min_valid = value < min_valid ? value : min_valid;
                max_valid = value > max_valid ? value : max_valid;
            }
        }
        const bool has_valid = min_valid <= max_valid;
        const float range = has_valid ? max_valid - min_valid : 0.f;
        std::vector<uint32_t> image((size_t)width * height, 0xFF000000u);
        for(int pixel_i = 0; pixel_i < width * height; pixel_i++){
            const float value = depth[pixel_i];
            uint8_t gray = 0;
            if(has_valid && value > 0.f && value < valid_max_depth){
                float normalized = (value - min_valid) / (range + 1e-6f);
                normalized = normalized < 0.f ? 0.f : (normalized > 1.f ? 1.f : normalized);
                gray = (uint8_t)(normalized * 255.f);
            }
            image[pixel_i] = (0xFFu << 24) | ((uint32_t)gray << 16) | ((uint32_t)gray << 8) | (uint32_t)gray;
        }
        return write_camera_png(path, image.data(), width, height);
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

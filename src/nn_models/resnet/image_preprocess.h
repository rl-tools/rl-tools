#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <random>

struct DecodedImage {
    uint8_t* pixels = nullptr;
    int width = 0, height = 0;
    bool valid = false;
    ~DecodedImage();
    DecodedImage();
    DecodedImage(DecodedImage&& o) noexcept;
    DecodedImage(const DecodedImage&) = delete;
    DecodedImage& operator=(const DecodedImage&) = delete;
};

DecodedImage decode_jpeg(const uint8_t* jpeg_bytes, size_t jpeg_size);
void random_resize_crop(const DecodedImage& src, uint8_t* dst, unsigned int target_size,
    float crop_scale_min, float crop_scale_max, float crop_ratio_min, float crop_ratio_max,
    std::mt19937& rng);
void center_crop_resize(const DecodedImage& src, uint8_t* dst, unsigned int target_size, std::vector<uint8_t>& resize_scratch);

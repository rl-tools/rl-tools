#include "image_preprocess.h"
#include <jpeglib.h>
#include <setjmp.h>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

DecodedImage::~DecodedImage() { std::free(pixels); }
DecodedImage::DecodedImage() = default;
DecodedImage::DecodedImage(DecodedImage&& o) noexcept : pixels(o.pixels), width(o.width), height(o.height), valid(o.valid) { o.pixels = nullptr; }

DecodedImage decode_jpeg(const uint8_t* jpeg_bytes, size_t jpeg_size) {
    DecodedImage img;
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jerr.error_exit = [](j_common_ptr cinfo) { longjmp(*static_cast<jmp_buf*>(cinfo->client_data), 1); };
    jmp_buf jmpbuf;
    cinfo.client_data = &jmpbuf;
    if (setjmp(jmpbuf)) {
        jpeg_destroy_decompress(&cinfo);
        return img;
    }
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, jpeg_bytes, jpeg_size);
    jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);
    img.width = cinfo.output_width;
    img.height = cinfo.output_height;
    int row_stride = img.width * 3;
    img.pixels = static_cast<uint8_t*>(std::malloc(img.width * img.height * 3));
    if (!img.pixels) { jpeg_destroy_decompress(&cinfo); return img; }
    while (cinfo.output_scanline < cinfo.output_height) {
        uint8_t* row = img.pixels + cinfo.output_scanline * row_stride;
        jpeg_read_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    img.valid = true;
    return img;
}

void random_resize_crop(const DecodedImage& src, uint8_t* dst, unsigned int target_size,
    float crop_scale_min, float crop_scale_max, float crop_ratio_min, float crop_ratio_max,
    std::mt19937& rng)
{
    std::uniform_real_distribution<float> scale_dist(crop_scale_min, crop_scale_max);
    std::uniform_real_distribution<float> ratio_dist(std::log(crop_ratio_min), std::log(crop_ratio_max));
    int crop_w, crop_h, crop_x, crop_y;
    bool found = false;
    for (int attempt = 0; attempt < 10; attempt++) {
        float area = static_cast<float>(src.width * src.height);
        float target_area = scale_dist(rng) * area;
        float ratio = std::exp(ratio_dist(rng));
        crop_w = static_cast<int>(std::sqrt(target_area * ratio) + 0.5f);
        crop_h = static_cast<int>(std::sqrt(target_area / ratio) + 0.5f);
        if (crop_w > 0 && crop_w <= src.width && crop_h > 0 && crop_h <= src.height) {
            std::uniform_int_distribution<int> x_dist(0, src.width - crop_w);
            std::uniform_int_distribution<int> y_dist(0, src.height - crop_h);
            crop_x = x_dist(rng); crop_y = y_dist(rng);
            found = true; break;
        }
    }
    if (!found) {
        float img_ratio = static_cast<float>(src.width) / src.height;
        if (img_ratio < crop_ratio_min) { crop_w = src.width; crop_h = static_cast<int>(src.width / crop_ratio_min); }
        else if (img_ratio > crop_ratio_max) { crop_h = src.height; crop_w = static_cast<int>(src.height * crop_ratio_max); }
        else { crop_w = src.width; crop_h = src.height; }
        crop_w = std::min(crop_w, src.width); crop_h = std::min(crop_h, src.height);
        crop_x = (src.width - crop_w) / 2; crop_y = (src.height - crop_h) / 2;
    }
    const uint8_t* crop_ptr = src.pixels + (crop_y * src.width + crop_x) * 3;
    stbir_resize_uint8_linear(crop_ptr, crop_w, crop_h, src.width * 3, dst, target_size, target_size, 0, STBIR_RGB);
}

void center_crop_resize(const DecodedImage& src, uint8_t* dst, unsigned int target_size, std::vector<uint8_t>& resize_scratch) {
    constexpr int RESIZE_SIZE = 256;
    float scale = static_cast<float>(RESIZE_SIZE) / std::min(src.width, src.height);
    int resized_w = std::max(static_cast<int>(src.width * scale + 0.5f), RESIZE_SIZE);
    int resized_h = std::max(static_cast<int>(src.height * scale + 0.5f), RESIZE_SIZE);
    size_t needed = resized_w * resized_h * 3;
    if (resize_scratch.size() < needed) resize_scratch.resize(needed);
    stbir_resize_uint8_linear(src.pixels, src.width, src.height, 0, resize_scratch.data(), resized_w, resized_h, 0, STBIR_RGB);
    int x_off = (resized_w - static_cast<int>(target_size)) / 2;
    int y_off = (resized_h - static_cast<int>(target_size)) / 2;
    for (unsigned int y = 0; y < target_size; y++)
        std::memcpy(dst + y * target_size * 3, resize_scratch.data() + (y_off + y) * resized_w * 3 + x_off * 3, target_size * 3);
}

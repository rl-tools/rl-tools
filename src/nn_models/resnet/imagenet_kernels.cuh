#pragma once

#include "imagenet_pipeline.h"

__constant__ float d_imagenet_mean[3] = {0.485f, 0.456f, 0.406f};
__constant__ float d_imagenet_std[3]  = {0.229f, 0.224f, 0.225f};

__device__ uint32_t xorshift32(uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
__device__ float rand_uniform(uint32_t& s) { return (float)(xorshift32(s) & 0x7FFFFFFF) / (float)0x7FFFFFFF; }

template<typename T>
__device__ void bilinear_normalize(const uint8_t* src, int pitch, int w, int h, float sx, float sy, T* dst) {
    int x0 = (int)floorf(sx), y0 = (int)floorf(sy), x1 = x0 + 1, y1 = y0 + 1;
    float fx = sx - (float)x0, fy = sy - (float)y0;
    x0 = max(0, min(x0, w-1)); x1 = max(0, min(x1, w-1));
    y0 = max(0, min(y0, h-1)); y1 = max(0, min(y1, h-1));
    for (int c = 0; c < 3; c++) {
        float v = (1.f-fx)*(1.f-fy)*(float)src[y0*pitch + x0*3 + c]
                + fx*(1.f-fy)*(float)src[y0*pitch + x1*3 + c]
                + (1.f-fx)*fy*(float)src[y1*pitch + x0*3 + c]
                + fx*fy*(float)src[y1*pitch + x1*3 + c];
        dst[c] = (T)((v / 255.f - d_imagenet_mean[c]) / d_imagenet_std[c]);
    }
}

template<typename T>
__global__ void train_crop_normalize(
    const uint8_t* __restrict__ pool, const ImageInfo* __restrict__ info, T* __restrict__ out,
    int TGT, float s_min, float s_max, float lr_min, float lr_max, float flip_p
) {
    int n = blockIdx.z, oy = blockIdx.y * blockDim.y + threadIdx.y, ox = blockIdx.x * blockDim.x + threadIdx.x;
    if (oy >= TGT || ox >= TGT) return;
    T* dst = out + ((size_t)n * TGT * TGT + oy * TGT + ox) * 3;
    const ImageInfo& im = info[n];
    if (im.img_w <= 0) { dst[0] = dst[1] = dst[2] = (T)0.f; return; }

    uint32_t rng = im.rng_seed;
    float area = (float)(im.img_w * im.img_h);
    int cx = 0, cy = 0, cw = im.img_w, ch = im.img_h;
    for (int a = 0; a < 10; a++) {
        float ta = (s_min + rand_uniform(rng) * (s_max - s_min)) * area;
        float ratio = expf(lr_min + rand_uniform(rng) * (lr_max - lr_min));
        int tw = (int)(sqrtf(ta * ratio) + 0.5f), th = (int)(sqrtf(ta / ratio) + 0.5f);
        if (tw > 0 && tw <= im.img_w && th > 0 && th <= im.img_h) {
            cx = (int)(rand_uniform(rng) * (float)(im.img_w - tw));
            cy = (int)(rand_uniform(rng) * (float)(im.img_h - th));
            cw = tw; ch = th; goto crop_done;
        }
        rand_uniform(rng); rand_uniform(rng);
    }
    { float r = (float)im.img_w / (float)im.img_h;
      if (r < expf(lr_min)) { cw = im.img_w; ch = (int)((float)im.img_w / expf(lr_min)); }
      else if (r > expf(lr_max)) { ch = im.img_h; cw = (int)((float)im.img_h * expf(lr_max)); }
      cw = min(cw, im.img_w); ch = min(ch, im.img_h);
      cx = (im.img_w - cw) / 2; cy = (im.img_h - ch) / 2; }
crop_done:;
    int hflip = rand_uniform(rng) < flip_p;
    float sx = hflip ? (float)cx + ((float)(TGT-1-ox) + 0.5f) * (float)cw / (float)TGT - 0.5f
                     : (float)cx + ((float)ox + 0.5f) * (float)cw / (float)TGT - 0.5f;
    float sy = (float)cy + ((float)oy + 0.5f) * (float)ch / (float)TGT - 0.5f;
    bilinear_normalize(pool + (size_t)n * DECODE_IMG_STRIDE, DECODE_PITCH, im.img_w, im.img_h, sx, sy, dst);
}

template<typename T>
__global__ void val_crop_normalize(
    const uint8_t* __restrict__ pool, const ImageInfo* __restrict__ info, T* __restrict__ out, int TGT
) {
    int n = blockIdx.z, oy = blockIdx.y * blockDim.y + threadIdx.y, ox = blockIdx.x * blockDim.x + threadIdx.x;
    if (oy >= TGT || ox >= TGT) return;
    T* dst = out + ((size_t)n * TGT * TGT + oy * TGT + ox) * 3;
    const ImageInfo& im = info[n];
    if (im.img_w <= 0) { dst[0] = dst[1] = dst[2] = (T)0.f; return; }
    float scale = 256.f / (float)min(im.img_w, im.img_h);
    float cw = (float)TGT / scale, ch = (float)TGT / scale;
    float cx_f = ((float)im.img_w - cw) * 0.5f, cy_f = ((float)im.img_h - ch) * 0.5f;
    float sx = cx_f + ((float)ox + 0.5f) * cw / (float)TGT - 0.5f;
    float sy = cy_f + ((float)oy + 0.5f) * ch / (float)TGT - 0.5f;
    bilinear_normalize(pool + (size_t)n * DECODE_IMG_STRIDE, DECODE_PITCH, im.img_w, im.img_h, sx, sy, dst);
}

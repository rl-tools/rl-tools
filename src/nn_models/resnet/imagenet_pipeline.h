#pragma once

#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <cmath>
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// --- CUDA / nvJPEG error helpers ---

static inline void require_cuda(cudaError_t status, const char* call){
    if(status != cudaSuccess){
        std::cerr << "CUDA call \"" << call << "\" failed: " << cudaGetErrorString(status) << std::endl;
        std::exit(1);
    }
}
static inline void require_nvjpeg(nvjpegStatus_t status, const char* call){
    if(status != NVJPEG_STATUS_SUCCESS){
        std::cerr << "nvJPEG call \"" << call << "\" failed with status " << static_cast<int>(status) << std::endl;
        std::exit(1);
    }
}
static inline void require_kernel_ok(cudaStream_t stream, const char* call){
    require_cuda(cudaGetLastError(), call);
#ifdef RL_TOOLS_DEBUG_CUDA_SYNC
    require_cuda(cudaStreamSynchronize(stream), call);
#else
    (void)stream;
#endif
}

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) require_cuda((call), #call)
#endif
#ifndef NVJPEG_CHECK
#define NVJPEG_CHECK(call) require_nvjpeg((call), #call)
#endif
#ifndef CUDA_KERNEL_CHECK
#define CUDA_KERNEL_CHECK(stream, call_name) require_kernel_ok((stream), (call_name))
#endif

// --- Decode pipeline constants ---

constexpr int NVJPEG_MAX_DIM = 2048;
constexpr int DECODE_PITCH = NVJPEG_MAX_DIM * 3;
constexpr size_t DECODE_IMG_STRIDE = (size_t)NVJPEG_MAX_DIM * NVJPEG_MAX_DIM * 3;

// --- Data structures ---

struct ImageInfo { int img_w, img_h; uint32_t rng_seed; };
struct ImgDims { int w, h; };

template<typename U>
struct TSQueue {
    std::queue<U> q;
    std::mutex mtx;
    std::condition_variable cv;
    void push(U v) { { std::lock_guard<std::mutex> lk(mtx); q.push(v); } cv.notify_one(); }
    U pop() { std::unique_lock<std::mutex> lk(mtx); cv.wait(lk, [&]{ return !q.empty(); }); U v = q.front(); q.pop(); return v; }
};

struct Sample {
    const uint8_t* image_data;
    size_t image_size;
    int64_t label;
};

template<typename TI, TI T_BATCH_SIZE>
struct DecodeSlot {
    static constexpr TI BATCH_SIZE = T_BATCH_SIZE;
    uint8_t* decode_pool;
    ImageInfo* gpu_info;
    TI* gpu_labels;
    cudaEvent_t ready_event;
    cudaEvent_t consumed_event;
    ImageInfo* cpu_info;
    TI* cpu_labels;
    const uint8_t* jpeg_data[T_BATCH_SIZE];
    size_t jpeg_size[T_BATCH_SIZE];
    nvjpegImage_t nj_dest[T_BATCH_SIZE];
};

template<int T_PIPES_PER_WORKER>
struct DecodeWorkerCtx {
    static constexpr int PIPES_PER_WORKER = T_PIPES_PER_WORKER;
    nvjpegHandle_t handle;
    nvjpegJpegDecoder_t decoder;
    nvjpegDecodeParams_t params;
    nvjpegJpegState_t states[T_PIPES_PER_WORKER];
    nvjpegBufferPinned_t pinned[T_PIPES_PER_WORKER];
    nvjpegBufferDevice_t device_buf[T_PIPES_PER_WORKER];
    nvjpegJpegStream_t jpeg_streams[T_PIPES_PER_WORKER];
    cudaStream_t cuda_stream;
};

// --- mmap-based binary dataset (produced by prepare_imagenet.py) ---

struct BinaryDataset {
    int fd = -1;
    uint8_t* mapped = nullptr;
    size_t file_size = 0;
    uint64_t num_samples = 0;
    struct IndexEntry { uint64_t offset; uint32_t size; uint32_t label; };
    const IndexEntry* index = nullptr;

    bool load(const std::string& path, bool mmap_populate) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            std::cerr << "Failed to open: " << path << ": " << std::strerror(errno) << " (errno " << errno << ")" << std::endl;
            return false;
        }
        struct stat st;
        if (fstat(fd, &st) < 0) {
            std::cerr << "fstat failed for: " << path << ": " << std::strerror(errno) << " (errno " << errno << ")" << std::endl;
            ::close(fd);
            fd = -1;
            return false;
        }
        file_size = st.st_size;
        int mmap_flags = MAP_PRIVATE;
        if (mmap_populate) {
            mmap_flags |= MAP_POPULATE;
        }
        mapped = static_cast<uint8_t*>(mmap(nullptr, file_size, PROT_READ, mmap_flags, fd, 0));
        if (mapped == MAP_FAILED) {
            const int mmap_errno = errno;
            std::cerr << "mmap failed for: " << path << ": " << std::strerror(mmap_errno) << " (errno " << mmap_errno << ")";
            if (mmap_errno == ENOMEM && mmap_populate) {
                std::cerr << " while using MAP_POPULATE";
            }
            std::cerr << std::endl;
            ::close(fd);
            fd = -1;
            mapped = nullptr;
            return false;
        }
        if (madvise(mapped, file_size, MADV_RANDOM) != 0) {
            std::cerr << "Warning: madvise(MADV_RANDOM) failed for: " << path << ": " << std::strerror(errno) << " (errno " << errno << ")" << std::endl;
        }
        num_samples = *reinterpret_cast<const uint64_t*>(mapped);
        index = reinterpret_cast<const IndexEntry*>(mapped + 8);
        return true;
    }
    void populate_samples(std::vector<Sample>& out) const {
        out.resize(num_samples);
        for (uint64_t i = 0; i < num_samples; i++)
            out[i] = {mapped + index[i].offset, index[i].size, static_cast<int64_t>(index[i].label)};
    }
    ~BinaryDataset() {
        if (mapped && mapped != MAP_FAILED) munmap(mapped, file_size);
        if (fd >= 0) ::close(fd);
    }
    BinaryDataset() = default;
    BinaryDataset(const BinaryDataset&) = delete;
    BinaryDataset& operator=(const BinaryDataset&) = delete;
};

// --- JPEG dimension parsing ---

inline ImgDims jpeg_dimensions(const uint8_t* data, size_t len) {
    if(data == nullptr || len < 4){
        return {0, 0};
    }
    if(data[0] != 0xFF || data[1] != 0xD8){
        return {0, 0};
    }
    auto is_sof = [](uint8_t marker){
        switch(marker){
            case 0xC0: case 0xC1: case 0xC2: case 0xC3:
            case 0xC5: case 0xC6: case 0xC7:
            case 0xC9: case 0xCA: case 0xCB:
            case 0xCD: case 0xCE: case 0xCF:
                return true;
            default:
                return false;
        }
    };

    size_t pos = 2;
    while(pos + 1 < len){
        while(pos < len && data[pos] != 0xFF){
            pos++;
        }
        if(pos + 1 >= len){
            break;
        }
        while(pos < len && data[pos] == 0xFF){
            pos++;
        }
        if(pos >= len){
            break;
        }
        uint8_t marker = data[pos++];

        if(marker == 0xD8 || marker == 0xD9 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)){
            if(marker == 0xD9){
                break;
            }
            continue;
        }
        if(marker == 0xDA){
            break;
        }
        if(pos + 1 >= len){
            break;
        }
        uint16_t seg_len = ((uint16_t)data[pos] << 8) | data[pos + 1];
        if(seg_len < 2){
            return {0, 0};
        }
        size_t seg_payload = (size_t)seg_len - 2;
        pos += 2;
        if(pos + seg_payload > len){
            return {0, 0};
        }
        if(is_sof(marker)){
            if(seg_payload < 5){
                return {0, 0};
            }
            int h = ((int)data[pos + 1] << 8) | data[pos + 2];
            int w = ((int)data[pos + 3] << 8) | data[pos + 4];
            if(w > 0 && h > 0){
                return {w, h};
            }
            return {0, 0};
        }
        pos += seg_payload;
    }
    return {0, 0};
}

// --- LR schedule ---

template<typename TI>
inline float cosine_lr(TI epoch, TI total_epochs, float base_lr, float min_lr, TI warmup_epochs, float warmup_lr) {
    if (epoch < warmup_epochs)
        return warmup_lr + (base_lr - warmup_lr) * static_cast<float>(epoch) / static_cast<float>(warmup_epochs);
    float progress = static_cast<float>(epoch - warmup_epochs) / static_cast<float>(total_epochs - warmup_epochs);
    return min_lr + (base_lr - min_lr) * 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * progress));
}

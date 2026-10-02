#include "golden_io.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

namespace {
    namespace fs = std::filesystem;

    class GoldenIoTest: public ::testing::Test{
    protected:
        fs::path directory;

        void SetUp() override{
            const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
            directory = fs::temp_directory_path()
                      / "rl_tools_golden_io_tests"
                      / (std::string(test_info->test_suite_name()) + "_" + test_info->name());
            std::error_code error;
            fs::remove_all(directory, error);
            error.clear();
            ASSERT_TRUE(fs::create_directories(directory, error));
            ASSERT_FALSE(error);
        }

        void TearDown() override{
            std::error_code error;
            fs::remove_all(directory, error);
        }

        std::string path(const std::string& filename) const{
            return (directory / filename).string();
        }
    };

    std::vector<unsigned char> read_bytes(const std::string& path){
        std::ifstream stream(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    }

    void write_bytes(const std::string& path, const std::vector<unsigned char>& bytes){
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(stream.is_open());
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        ASSERT_TRUE(stream.good());
    }

    void store_u32_le(std::vector<unsigned char>& bytes, size_t offset, uint32_t value){
        ASSERT_LE(offset + 4, bytes.size());
        for(size_t byte_i = 0; byte_i < 4; byte_i++){
            bytes[offset + byte_i] = static_cast<unsigned char>((value >> (byte_i * 8)) & 0xFFu);
        }
    }

    void store_u64_le(std::vector<unsigned char>& bytes, size_t offset, uint64_t value){
        ASSERT_LE(offset + 8, bytes.size());
        for(size_t byte_i = 0; byte_i < 8; byte_i++){
            bytes[offset + byte_i] = static_cast<unsigned char>((value >> (byte_i * 8)) & 0xFFu);
        }
    }

    void expect_uint32_load_failure(const std::string& path, int cameras = 4, int width = 3, int height = 2){
        const std::vector<uint32_t> sentinel = {19u, 23u, 29u};
        std::vector<uint32_t> output = sentinel;
        EXPECT_FALSE(golden::load_multi_camera_uint32_bin(path, cameras, width, height, output));
        EXPECT_EQ(output, sentinel);
    }

    void expect_float_load_failure(const std::string& path, int cameras = 4, int width = 3, int height = 2){
        const std::vector<float> sentinel = {-3.5f, 8.25f};
        std::vector<float> output = sentinel;
        EXPECT_FALSE(golden::load_multi_camera_float_bin(path, cameras, width, height, output));
        EXPECT_EQ(output, sentinel);
    }
}

TEST_F(GoldenIoTest, MultiCameraUint32RoundTrip){
    constexpr int NUM_CAMERAS = 4;
    constexpr int WIDTH = 3;
    constexpr int HEIGHT = 2;
    std::vector<uint32_t> expected(NUM_CAMERAS * WIDTH * HEIGHT);
    for(size_t value_i = 0; value_i < expected.size(); value_i++){
        expected[value_i] = static_cast<uint32_t>(value_i * 0x01020304u);
    }
    expected.front() = 0u;
    expected.back() = std::numeric_limits<uint32_t>::max();

    const std::string binary_path = path("uint32.bin");
    ASSERT_TRUE(golden::write_multi_camera_uint32_bin(binary_path, expected.data(), NUM_CAMERAS, WIDTH, HEIGHT));
    std::vector<uint32_t> actual;
    ASSERT_TRUE(golden::load_multi_camera_uint32_bin(binary_path, NUM_CAMERAS, WIDTH, HEIGHT, actual));
    EXPECT_EQ(actual, expected);
}

TEST_F(GoldenIoTest, MultiCameraFloatRoundTrip){
    constexpr int NUM_CAMERAS = 4;
    constexpr int WIDTH = 3;
    constexpr int HEIGHT = 2;
    const std::vector<float> expected = {
        0.0f, -0.0f, 0.25f, -1.5f, 2.75f, 1000.125f,
        -0.03125f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f,
        1.0e-10f, -1.0e-10f, 3.1415927f, -2.7182817f, 0.5f, -0.5f,
        7.25f, 9.5f, 11.75f, 13.0f, 17.5f, 19.25f
    };

    const std::string binary_path = path("float.bin");
    ASSERT_TRUE(golden::write_multi_camera_float_bin(binary_path, expected.data(), NUM_CAMERAS, WIDTH, HEIGHT));
    std::vector<float> actual;
    ASSERT_TRUE(golden::load_multi_camera_float_bin(binary_path, NUM_CAMERAS, WIDTH, HEIGHT, actual));
    ASSERT_EQ(actual.size(), expected.size());
    EXPECT_EQ(
        std::vector<unsigned char>(reinterpret_cast<const unsigned char*>(actual.data()),
                                   reinterpret_cast<const unsigned char*>(actual.data() + actual.size())),
        std::vector<unsigned char>(reinterpret_cast<const unsigned char*>(expected.data()),
                                   reinterpret_cast<const unsigned char*>(expected.data() + expected.size()))
    );
}

TEST_F(GoldenIoTest, RejectsMalformedHeaderWithoutChangingOutput){
    constexpr int NUM_CAMERAS = 4;
    constexpr int WIDTH = 3;
    constexpr int HEIGHT = 2;
    std::vector<uint32_t> values(NUM_CAMERAS * WIDTH * HEIGHT, 7u);
    const std::string valid_path = path("valid.bin");
    ASSERT_TRUE(golden::write_multi_camera_uint32_bin(valid_path, values.data(), NUM_CAMERAS, WIDTH, HEIGHT));
    const std::vector<unsigned char> valid = read_bytes(valid_path);
    ASSERT_GT(valid.size(), golden::detail::MULTI_CAMERA_BINARY_HEADER_SIZE);

    struct Mutation{
        const char* filename;
        size_t offset;
        uint32_t value;
    };
    const Mutation mutations[] = {
        {"bad_version.bin", 8, golden::MULTI_CAMERA_BINARY_VERSION + 1},
        {"bad_header_size.bin", 12, golden::detail::MULTI_CAMERA_BINARY_HEADER_SIZE + 4},
        {"bad_type.bin", 16, 99},
        {"bad_camera_count.bin", 20, 0},
        {"bad_height.bin", 24, 0},
        {"bad_width.bin", 28, 0},
        {"bad_channels.bin", 32, 0}
    };
    for(const auto& mutation: mutations){
        SCOPED_TRACE(mutation.filename);
        std::vector<unsigned char> bytes = valid;
        store_u32_le(bytes, mutation.offset, mutation.value);
        const std::string corrupt_path = path(mutation.filename);
        write_bytes(corrupt_path, bytes);
        expect_uint32_load_failure(corrupt_path);
    }

    std::vector<unsigned char> bad_magic = valid;
    bad_magic[0] ^= 0xFFu;
    const std::string bad_magic_path = path("bad_magic.bin");
    write_bytes(bad_magic_path, bad_magic);
    expect_uint32_load_failure(bad_magic_path);

    std::vector<unsigned char> bad_count = valid;
    store_u64_le(bad_count, 36, values.size() + 1);
    const std::string bad_count_path = path("bad_count.bin");
    write_bytes(bad_count_path, bad_count);
    expect_uint32_load_failure(bad_count_path);

    std::vector<unsigned char> bad_payload_size = valid;
    store_u64_le(bad_payload_size, 44, valid.size() - golden::detail::MULTI_CAMERA_BINARY_HEADER_SIZE + 1);
    const std::string bad_payload_size_path = path("bad_payload_size.bin");
    write_bytes(bad_payload_size_path, bad_payload_size);
    expect_uint32_load_failure(bad_payload_size_path);

    std::vector<unsigned char> corrupt_stream = valid;
    corrupt_stream[golden::detail::MULTI_CAMERA_BINARY_HEADER_SIZE + 2] ^= 0xFFu;
    const std::string corrupt_stream_path = path("corrupt_stream.bin");
    write_bytes(corrupt_stream_path, corrupt_stream);
    expect_uint32_load_failure(corrupt_stream_path);

    std::vector<unsigned char> wrong_known_type = valid;
    store_u32_le(wrong_known_type, 16, static_cast<uint32_t>(golden::MultiCameraElementType::FLOAT32));
    const std::string wrong_known_type_path = path("wrong_known_type.bin");
    write_bytes(wrong_known_type_path, wrong_known_type);
    expect_uint32_load_failure(wrong_known_type_path);

    expect_uint32_load_failure(valid_path, NUM_CAMERAS, WIDTH + 1, HEIGHT);
    expect_float_load_failure(valid_path);
}

TEST_F(GoldenIoTest, RejectsTruncatedAndTrailingPayloadWithoutChangingOutput){
    constexpr int NUM_CAMERAS = 4;
    constexpr int WIDTH = 3;
    constexpr int HEIGHT = 2;
    std::vector<float> values(NUM_CAMERAS * WIDTH * HEIGHT);
    for(size_t value_i = 0; value_i < values.size(); value_i++){
        values[value_i] = static_cast<float>(value_i) / 3.0f;
    }
    const std::string valid_path = path("valid_float.bin");
    ASSERT_TRUE(golden::write_multi_camera_float_bin(valid_path, values.data(), NUM_CAMERAS, WIDTH, HEIGHT));
    const std::vector<unsigned char> valid = read_bytes(valid_path);
    ASSERT_GT(valid.size(), golden::detail::MULTI_CAMERA_BINARY_HEADER_SIZE);

    std::vector<unsigned char> truncated = valid;
    truncated.pop_back();
    const std::string truncated_path = path("truncated.bin");
    write_bytes(truncated_path, truncated);
    expect_float_load_failure(truncated_path);

    std::vector<unsigned char> trailing = valid;
    trailing.push_back(0xA5u);
    const std::string trailing_path = path("trailing.bin");
    write_bytes(trailing_path, trailing);
    expect_float_load_failure(trailing_path);
}

TEST_F(GoldenIoTest, MultiCameraHeaderAndPayloadBytesArePinned){
    constexpr int NUM_CAMERAS = 4;
    constexpr int WIDTH = 3;
    constexpr int HEIGHT = 2;
    std::vector<uint32_t> values(NUM_CAMERAS * WIDTH * HEIGHT);
    for(int camera_i = 0; camera_i < NUM_CAMERAS; camera_i++){
        for(int row = 0; row < HEIGHT; row++){
            for(int column = 0; column < WIDTH; column++){
                values[(size_t)(camera_i * HEIGHT + row) * WIDTH + column] = static_cast<uint32_t>(camera_i * 100 + row * 10 + column);
            }
        }
    }
    const std::string binary_path = path("pinned.bin");
    ASSERT_TRUE(golden::write_multi_camera_uint32_bin(binary_path, values.data(), NUM_CAMERAS, WIDTH, HEIGHT));
    const std::vector<unsigned char> bytes = read_bytes(binary_path);
    ASSERT_GT(bytes.size(), golden::detail::MULTI_CAMERA_BINARY_HEADER_SIZE);
    const size_t payload_bytes = bytes.size() - golden::detail::MULTI_CAMERA_BINARY_HEADER_SIZE;

    unsigned char expected_header[golden::detail::MULTI_CAMERA_BINARY_HEADER_SIZE] = {
        'R', 'L', 'T', 'M', 'C', 'A', 'M', 0,
        3, 0, 0, 0,     // version
        52, 0, 0, 0,    // header size
        2, 0, 0, 0,     // element type: UINT32
        4, 0, 0, 0,     // num_cameras
        2, 0, 0, 0,     // height
        3, 0, 0, 0,     // width
        1, 0, 0, 0,     // channels
        24, 0, 0, 0, 0, 0, 0, 0, // element count
        0, 0, 0, 0, 0, 0, 0, 0   // payload bytes (deflated size, filled below)
    };
    for(int byte_i = 0; byte_i < 8; byte_i++){
        expected_header[44 + byte_i] = (unsigned char)((payload_bytes >> (8 * byte_i)) & 0xFFu);
    }
    EXPECT_EQ(std::memcmp(bytes.data(), expected_header, sizeof(expected_header)), 0);
    EXPECT_EQ(bytes[golden::detail::MULTI_CAMERA_BINARY_HEADER_SIZE], 0x78); // zlib stream header

    std::vector<char> words(values.size() * sizeof(uint32_t));
    ASSERT_EQ(stbi_zlib_decode_buffer(words.data(), (int)words.size(), reinterpret_cast<const char*>(bytes.data() + golden::detail::MULTI_CAMERA_BINARY_HEADER_SIZE), (int)payload_bytes), (int)words.size());
    for(int camera_i = 0; camera_i < NUM_CAMERAS; camera_i++){
        for(int row = 0; row < HEIGHT; row++){
            for(int column = 0; column < WIDTH; column++){
                const auto* word = reinterpret_cast<const unsigned char*>(words.data()) + ((size_t)(camera_i * HEIGHT + row) * WIDTH + column) * sizeof(uint32_t);
                const uint32_t value = (uint32_t)word[0] | ((uint32_t)word[1] << 8) | ((uint32_t)word[2] << 16) | ((uint32_t)word[3] << 24);
                EXPECT_EQ(value, static_cast<uint32_t>(camera_i * 100 + row * 10 + column))
                    << "camera " << camera_i << " row " << row << " column " << column;
            }
        }
    }
}

// multi-channel payloads interleave the channels per pixel; a channel mismatch is a load failure
TEST_F(GoldenIoTest, MultiCameraChannelsHeaderBytesArePinned){
    constexpr int NUM_CAMERAS = 4;
    constexpr int WIDTH = 3;
    constexpr int HEIGHT = 2;
    constexpr int CHANNELS = 2;
    std::vector<float> values((size_t)NUM_CAMERAS * WIDTH * HEIGHT * CHANNELS);
    for(size_t value_i = 0; value_i < values.size(); value_i++){
        values[value_i] = (float)value_i - 10.5f;
    }
    const std::string binary_path = path("pinned_channels.bin");
    ASSERT_TRUE(golden::write_multi_camera_float_bin(binary_path, values.data(), NUM_CAMERAS, WIDTH, HEIGHT, CHANNELS));
    const std::vector<unsigned char> bytes = read_bytes(binary_path);
    ASSERT_GT(bytes.size(), golden::detail::MULTI_CAMERA_BINARY_HEADER_SIZE);

    const unsigned char expected_header[44] = {
        'R', 'L', 'T', 'M', 'C', 'A', 'M', 0,
        3, 0, 0, 0,     // version
        52, 0, 0, 0,    // header size
        1, 0, 0, 0,     // element type: FLOAT32
        4, 0, 0, 0,     // num_cameras
        2, 0, 0, 0,     // height
        3, 0, 0, 0,     // width
        2, 0, 0, 0,     // channels
        48, 0, 0, 0, 0, 0, 0, 0 // element count
    };
    EXPECT_EQ(std::memcmp(bytes.data(), expected_header, sizeof(expected_header)), 0);

    std::vector<float> loaded;
    ASSERT_TRUE(golden::load_multi_camera_float_bin(binary_path, NUM_CAMERAS, WIDTH, HEIGHT, loaded, CHANNELS));
    EXPECT_EQ(loaded, values);
    EXPECT_FALSE(golden::load_multi_camera_float_bin(binary_path, NUM_CAMERAS, WIDTH, HEIGHT, loaded)); // channel mismatch must be rejected
}

TEST_F(GoldenIoTest, NormalEncodingIsPinned){
    const float miss[3] = {0.f, 0.f, 0.f};
    EXPECT_EQ(golden::normal_rgba(miss), golden::rgba(128, 128, 128));
    const float facing_camera[3] = {-1.f, 0.f, 0.f};
    EXPECT_EQ(golden::normal_rgba(facing_camera), golden::rgba(0, 128, 128));
    const float diagonal[3] = {0.57735f, -0.57735f, 0.57735f};
    EXPECT_EQ(golden::normal_rgba(diagonal), golden::rgba(201, 54, 201));
    const float out_of_range[3] = {2.f, -2.f, 1.f};
    EXPECT_EQ(golden::normal_rgba(out_of_range), golden::rgba(255, 0, 255));
}

// the segmentation false-color encoding is corpus surface in both the overlay and
// ProcTHOR suites: the segmentation review image is validated to be exactly this encoding of
// segmentation.bin, so the mapping must never drift
TEST_F(GoldenIoTest, SegmentationEncodingIsPinned){
    EXPECT_EQ(golden::segmentation_false_color(golden::SEGMENTATION_BACKGROUND_ID), golden::rgba(16, 16, 16));
    EXPECT_EQ(golden::segmentation_false_color(0u), golden::rgba(146, 255, 42));
    EXPECT_EQ(golden::segmentation_false_color(1u), golden::rgba(234, 38, 144));
    EXPECT_EQ(golden::segmentation_false_color(2u), golden::rgba(142, 243, 57));
    EXPECT_EQ(golden::segmentation_false_color(40u), golden::rgba(92, 35, 248));
    for(uint32_t instance_id = 0; instance_id < 4096; instance_id++){
        const uint32_t color = golden::segmentation_false_color(instance_id);
        ASSERT_EQ(color >> 24, 0xFFu) << "instance id " << instance_id;
        // every channel >= 32 keeps the background gray (16,16,16) unreachable for real ids
        ASSERT_GE(color & 0xFFu, 32u) << "instance id " << instance_id;
        ASSERT_GE((color >> 8) & 0xFFu, 32u) << "instance id " << instance_id;
        ASSERT_GE((color >> 16) & 0xFFu, 32u) << "instance id " << instance_id;
    }
}

// flow_status.png / flow_forward_status.png are corpus surface: the status codes map to fixed colors
TEST_F(GoldenIoTest, FlowStatusEncodingIsPinned){
    using FlowStatus = rl_tools::rendering::raytracing::FlowStatus;
    EXPECT_EQ(golden::flow_status_false_color(FlowStatus::VALID), golden::rgba(255, 255, 255));
    EXPECT_EQ(golden::flow_status_false_color(FlowStatus::MISS), golden::rgba(0, 0, 0));
    EXPECT_EQ(golden::flow_status_false_color(FlowStatus::BEHIND), golden::rgba(255, 0, 255));
    EXPECT_EQ(golden::flow_status_false_color(FlowStatus::OUT_OF_FRAME), golden::rgba(32, 64, 255));
    EXPECT_EQ(golden::flow_status_false_color(FlowStatus::OCCLUDED), golden::rgba(255, 144, 0));
    EXPECT_EQ(golden::flow_status_false_color(5u), golden::rgba(255, 0, 0));
    EXPECT_EQ(golden::flow_status_false_color(std::numeric_limits<uint32_t>::max()), golden::rgba(255, 0, 0));
    EXPECT_EQ(golden::flow_status_diff_color(FlowStatus::VALID, FlowStatus::VALID), golden::rgba(0, 0, 0));
    EXPECT_NE(golden::flow_status_diff_color(FlowStatus::VALID, FlowStatus::OCCLUDED), golden::rgba(0, 0, 0));
    const uint32_t status[4] = {FlowStatus::VALID, FlowStatus::MISS, FlowStatus::OCCLUDED, 9u};
    uint32_t image[4];
    golden::colorize_flow_status(status, 4, image);
    EXPECT_EQ(image[0], golden::rgba(255, 255, 255));
    EXPECT_EQ(image[1], golden::rgba(0, 0, 0));
    EXPECT_EQ(image[2], golden::rgba(255, 144, 0));
    EXPECT_EQ(image[3], golden::rgba(255, 0, 0));
}

TEST_F(GoldenIoTest, LayoutPathsAreStable){
    const std::string root = golden::layout::goldens_root("test-data");
    EXPECT_EQ(root, "test-data/rendering/raytracing/goldens");
    EXPECT_EQ(golden::layout::procthor_directory(root), root + "/procthor-train-1");
    EXPECT_EQ(golden::layout::procthor_view_directory(root, "07"), root + "/procthor-train-1/07");
    EXPECT_EQ(golden::layout::manifest_path(root + "/procthor-train-1"), root + "/procthor-train-1/manifest.json");
    EXPECT_EQ(golden::layout::view_path(root + "/procthor-train-1/07"), root + "/procthor-train-1/07/view.json");

    const std::string modality = golden::layout::output_directory(root + "/procthor-train-1/07", "camera-objects", "high-mb4");
    EXPECT_EQ(modality, root + "/procthor-train-1/07/camera-objects/high-mb4");
    EXPECT_EQ(golden::layout::golden_png(modality, 0), modality + "/golden-v00.png");
    EXPECT_EQ(golden::layout::golden_png(modality, 1), modality + "/golden-v01.png");
    EXPECT_EQ(golden::layout::golden_bin(modality, 12), modality + "/data/golden-v12.bin");
    EXPECT_EQ(golden::layout::render_png(modality, "vulkan-lavapipe"), modality + "/render-vulkan-lavapipe.png");
    EXPECT_EQ(golden::layout::diff_png(modality, golden::layout::golden_name(2), golden::layout::golden_name(1)), modality + "/diff-golden-v02-golden-v01.png");
    EXPECT_EQ(golden::layout::diff_png(modality, golden::layout::golden_name(2), "render-optix"), modality + "/diff-golden-v02-render-optix.png");
    EXPECT_LT(std::string("golden-v00.png"), std::string("golden-v01.png"));
    EXPECT_LT(std::string("golden-v12.png"), std::string("render-generic.png"));
}

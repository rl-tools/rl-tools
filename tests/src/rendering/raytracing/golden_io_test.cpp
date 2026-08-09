#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

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
    ASSERT_EQ(valid.size(), golden::detail::MULTI_CAMERA_BINARY_HEADER_SIZE + values.size() * sizeof(uint32_t));

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
        {"bad_width.bin", 28, 0}
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
    store_u64_le(bad_count, 32, values.size() + 1);
    const std::string bad_count_path = path("bad_count.bin");
    write_bytes(bad_count_path, bad_count);
    expect_uint32_load_failure(bad_count_path);

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

TEST_F(GoldenIoTest, FourCameraGridPngRoundTrip){
    constexpr int NUM_CAMERAS = 4;
    constexpr int WIDTH = 3;
    constexpr int HEIGHT = 2;
    std::vector<uint32_t> cameras(NUM_CAMERAS * WIDTH * HEIGHT);
    for(int camera_i = 0; camera_i < NUM_CAMERAS; camera_i++){
        for(int pixel_i = 0; pixel_i < WIDTH * HEIGHT; pixel_i++){
            cameras[static_cast<size_t>(camera_i) * WIDTH * HEIGHT + pixel_i] = golden::rgba(
                static_cast<uint8_t>(20 + camera_i * 40),
                static_cast<uint8_t>(30 + pixel_i * 15),
                static_cast<uint8_t>(200 - camera_i * 20 - pixel_i),
                static_cast<uint8_t>(100 + camera_i * 10 + pixel_i)
            );
        }
    }

    std::vector<uint32_t> grid;
    ASSERT_TRUE(golden::make_camera_grid(cameras.data(), NUM_CAMERAS, WIDTH, HEIGHT, grid));
    ASSERT_EQ(grid.size(), static_cast<size_t>(WIDTH * 2 * HEIGHT * 2));
    std::vector<uint32_t> split;
    ASSERT_TRUE(golden::split_camera_grid(grid.data(), NUM_CAMERAS, WIDTH, HEIGHT, split));
    EXPECT_EQ(split, cameras);

    const std::string png_path = path("grid.png");
    ASSERT_TRUE(golden::write_camera_grid_png(png_path, cameras.data(), NUM_CAMERAS, WIDTH, HEIGHT));
    std::vector<uint32_t> loaded;
    ASSERT_TRUE(golden::load_camera_grid_png(png_path, NUM_CAMERAS, WIDTH, HEIGHT, loaded));
    EXPECT_EQ(loaded, cameras);

    const std::vector<uint32_t> sentinel = {0x12345678u};
    loaded = sentinel;
    EXPECT_FALSE(golden::load_camera_grid_png(png_path, NUM_CAMERAS, WIDTH + 1, HEIGHT, loaded));
    EXPECT_EQ(loaded, sentinel);
}

TEST_F(GoldenIoTest, LayoutPathsAreStable){
    const std::string root = "golden-root";
    EXPECT_EQ(golden::layout::procthor_static_scene_directory(root), "golden-root/procthor_static_scene");
    EXPECT_EQ(golden::layout::procthor_pose_directory(root, "07"), "golden-root/procthor_static_scene/07");
    EXPECT_EQ(golden::layout::legacy_procthor_pose_directory(root, "07"), "golden-root/07");
    EXPECT_EQ(golden::layout::overlay_directory(root), "golden-root/overlay");
    EXPECT_EQ(golden::layout::overlay_manifest_path(root), "golden-root/overlay/manifest.json");

    const auto targets = golden::layout::scenario_target_paths(root, "partial_shared", "updated", "oblique");
    EXPECT_EQ(targets.directory, "golden-root/overlay/partial_shared/updated/oblique");
    EXPECT_EQ(targets.rgb_png, targets.directory + "/rgb.png");
    EXPECT_EQ(targets.depth_bin, targets.directory + "/depth.bin");
    EXPECT_EQ(targets.depth_png, targets.directory + "/depth.png");
    EXPECT_EQ(targets.segmentation_bin, targets.directory + "/segmentation.bin");
    EXPECT_EQ(targets.segmentation_png, targets.directory + "/segmentation.png");

    const auto reviews = golden::layout::scenario_review_paths(
        "artifact-root", "vulkan", "partial_shared", "updated", "oblique"
    );
    EXPECT_EQ(reviews.directory, "artifact-root/vulkan/partial_shared/updated/oblique");
    EXPECT_EQ(reviews.rgb_target_png, reviews.directory + "/rgb_target.png");
    EXPECT_EQ(reviews.rgb_current_png, reviews.directory + "/rgb_current.png");
    EXPECT_EQ(reviews.rgb_diff_png, reviews.directory + "/rgb_diff.png");
    EXPECT_EQ(reviews.depth_target_png, reviews.directory + "/depth_target.png");
    EXPECT_EQ(reviews.depth_current_png, reviews.directory + "/depth_current.png");
    EXPECT_EQ(reviews.depth_diff_png, reviews.directory + "/depth_diff.png");
    EXPECT_EQ(reviews.segmentation_target_png, reviews.directory + "/segmentation_target.png");
    EXPECT_EQ(reviews.segmentation_current_png, reviews.directory + "/segmentation_current.png");
    EXPECT_EQ(reviews.segmentation_diff_png, reviews.directory + "/segmentation_diff.png");
}

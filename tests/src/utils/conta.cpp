#include <conta/conta.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace{
    constexpr char ABC_SHA1[] = "a9993e364706816aba3e25717850c26c9cd0d89d";
    constexpr char EMPTY_SHA1[] = "da39a3ee5e6b4b0d3255bfef95601890afd80709";
    std::filesystem::path scratch_directory(){
        const ::testing::TestInfo* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::filesystem::path directory = std::filesystem::path(::testing::TempDir()) / "conta_test" / test_info->name();
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);
        return directory;
    }
    void write_file(const std::filesystem::path& path, const std::string& content){
        std::filesystem::create_directories(path.parent_path());
        std::ofstream file(path, std::ios::binary);
        file << content;
    }
}

TEST(UTILS_CONTA, SHA1_KNOWN_VECTORS){
    std::filesystem::path scratch = scratch_directory();
    write_file(scratch / "empty", "");
    write_file(scratch / "abc", "abc");
    std::string hex;
    ASSERT_TRUE(conta::detail::sha1_file(scratch / "empty", hex));
    ASSERT_EQ(hex, EMPTY_SHA1);
    ASSERT_TRUE(conta::detail::sha1_file(scratch / "abc", hex));
    ASSERT_EQ(hex, ABC_SHA1);
    ASSERT_FALSE(conta::detail::sha1_file(scratch / "missing", hex));
}

TEST(UTILS_CONTA, SHA1_MULTI_BLOCK){
    std::filesystem::path scratch = scratch_directory();
    std::string content(1000000, 'a');
    write_file(scratch / "million_a", content);
    std::string hex;
    ASSERT_TRUE(conta::detail::sha1_file(scratch / "million_a", hex));
    ASSERT_EQ(hex, "34aa973cd4c4daa4f61eeb2bdbad27316534016f");
}

TEST(UTILS_CONTA, HASH_VALIDATION){
    std::string hash = ABC_SHA1;
    ASSERT_TRUE(conta::detail::normalize_hash(hash));
    hash = "A9993E364706816ABA3E25717850C26C9CD0D89D";
    ASSERT_TRUE(conta::detail::normalize_hash(hash));
    ASSERT_EQ(hash, ABC_SHA1);
    hash = std::string(ABC_SHA1).substr(0, 39);
    ASSERT_FALSE(conta::detail::normalize_hash(hash));
    hash = std::string(ABC_SHA1) + "0";
    ASSERT_FALSE(conta::detail::normalize_hash(hash));
    hash = "g9993e364706816aba3e25717850c26c9cd0d89d";
    ASSERT_FALSE(conta::detail::normalize_hash(hash));
    hash = "../../../../../../../../etc/passwd00000000";
    ASSERT_FALSE(conta::detail::normalize_hash(hash));
    conta::Config config;
    config.root = "/nonexistent";
    std::string path, error;
    ASSERT_FALSE(conta::resolve(config, "conta:not-a-hash", path, error));
    ASSERT_NE(error.find("invalid hash"), std::string::npos);
}

TEST(UTILS_CONTA, ROOT_HIT){
    std::filesystem::path scratch = scratch_directory();
    conta::Config config;
    config.root = (scratch / "root").string();
    write_file(scratch / "root" / "data" / ABC_SHA1, "abc");
    std::string path, error;
    ASSERT_TRUE(conta::resolve(config, ABC_SHA1, path, error)) << error;
    ASSERT_EQ(path, (scratch / "root" / "data" / ABC_SHA1).string());
}

TEST(UTILS_CONTA, ROOT_MISS){
    std::filesystem::path scratch = scratch_directory();
    conta::Config config;
    config.root = (scratch / "root").string();
    std::string path, error;
    ASSERT_FALSE(conta::resolve(config, ABC_SHA1, path, error));
    ASSERT_NE(error.find("CONTA_ROOT"), std::string::npos);
    ASSERT_NE(error.find((scratch / "root" / "data" / ABC_SHA1).string()), std::string::npos);
}

TEST(UTILS_CONTA, ROOT_LFS_POINTER){
    std::filesystem::path scratch = scratch_directory();
    conta::Config config;
    config.root = (scratch / "root").string();
    write_file(scratch / "root" / "data" / ABC_SHA1, "version https://git-lfs.github.com/spec/v1\noid sha256:0000000000000000000000000000000000000000000000000000000000000000\nsize 123\n");
    std::string path, error;
    ASSERT_FALSE(conta::resolve(config, ABC_SHA1, path, error));
    ASSERT_NE(error.find("git lfs pull"), std::string::npos);
}

TEST(UTILS_CONTA, CACHE_HIT_NO_NETWORK){
    std::filesystem::path scratch = scratch_directory();
    conta::Config config;
    config.cache = (scratch / "cache").string();
    config.url_base = "http://127.0.0.1:1/";
    write_file(scratch / "cache" / ABC_SHA1, "abc");
    std::string path, error;
    ASSERT_TRUE(conta::resolve(config, ABC_SHA1, path, error)) << error;
    ASSERT_EQ(path, (scratch / "cache" / ABC_SHA1).string());
}

TEST(UTILS_CONTA, DOWNLOAD_FILE_URL){
    std::filesystem::path scratch = scratch_directory();
    conta::Config config;
    config.cache = (scratch / "cache").string();
    config.url_base = "file://" + (scratch / "store").string() + "/";
    write_file(scratch / "store" / ABC_SHA1, "abc");
    std::string path, error;
    ASSERT_TRUE(conta::resolve(config, ABC_SHA1, path, error)) << error;
    ASSERT_EQ(path, (scratch / "cache" / ABC_SHA1).string());
    std::string hex;
    ASSERT_TRUE(conta::detail::sha1_file(path, hex));
    ASSERT_EQ(hex, ABC_SHA1);
}

TEST(UTILS_CONTA, DOWNLOAD_HASH_MISMATCH){
    std::filesystem::path scratch = scratch_directory();
    conta::Config config;
    config.cache = (scratch / "cache").string();
    config.url_base = "file://" + (scratch / "store").string() + "/";
    write_file(scratch / "store" / ABC_SHA1, "not abc");
    std::string path, error;
    ASSERT_FALSE(conta::resolve(config, ABC_SHA1, path, error));
    ASSERT_NE(error.find("hash mismatch"), std::string::npos);
    ASSERT_FALSE(std::filesystem::exists(scratch / "cache" / ABC_SHA1));
}

TEST(UTILS_CONTA, BATCH_ALL_OR_NOTHING){
    std::filesystem::path scratch = scratch_directory();
    conta::Config config;
    config.cache = (scratch / "cache").string();
    config.url_base = "file://" + (scratch / "store").string() + "/";
    write_file(scratch / "store" / ABC_SHA1, "abc");
    std::vector<std::string> hashes = {ABC_SHA1, EMPTY_SHA1};
    std::vector<std::string> paths;
    std::string error;
    ASSERT_FALSE(conta::resolve(config, hashes, paths, error));
    ASSERT_TRUE(paths.empty());
    hashes = {ABC_SHA1};
    ASSERT_TRUE(conta::resolve(config, hashes, paths, error)) << error;
    ASSERT_EQ(paths.size(), 1u);
}

TEST(UTILS_CONTA, NETWORK_DOWNLOAD){
    const char* enabled = std::getenv("RL_TOOLS_TEST_CONTA_NETWORK");
    if(enabled == nullptr || std::string(enabled) != "1"){
        GTEST_SKIP() << "set RL_TOOLS_TEST_CONTA_NETWORK=1 to enable";
    }
    std::filesystem::path scratch = scratch_directory();
    conta::Config config;
    config.cache = (scratch / "cache").string();
    config.url_base = conta::DEFAULT_URL_BASE;
    constexpr char BLOB[] = "a38b2994e7674f467fe81e86d2f21c45bfd965a0";
    std::string path, error;
    ASSERT_TRUE(conta::resolve(config, BLOB, path, error)) << error;
    std::string hex;
    ASSERT_TRUE(conta::detail::sha1_file(path, hex));
    ASSERT_EQ(hex, BLOB);
}

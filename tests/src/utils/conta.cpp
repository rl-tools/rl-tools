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
    constexpr char ABC_SHA256[] = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    constexpr char EMPTY_SHA256[] = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    constexpr char HELLO_SHA1[] = "aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d";
    constexpr char HELLO_SHA256[] = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
    std::string read_text(const std::filesystem::path& path){
        std::string text;
        conta::detail::read_file(path, text);
        return text;
    }
    std::string file_url(const std::filesystem::path& path){
        return "file://" + path.string();
    }
    std::string entry(const std::string& sha1, const std::string& sha256, const std::string& shards = "\"local\""){
        return "{\"sha1\": \"" + sha1 + "\", \"sha256\": \"" + sha256 + "\", \"shards\": [" + shards + "], \"description\": \"test blob\"}";
    }
    std::string index_text(const std::string& shards, const std::string& entries){
        return "{\n\"shards\": {\n" + shards + "\n},\n\"entries\": [\n" + entries + "\n]\n}\n";
    }
    std::string local_shard(const std::filesystem::path& store){
        return "\"local\": {\"type\": \"http\", \"url\": \"" + file_url(store) + "/{sha1}\"}";
    }
    std::string publish(const std::filesystem::path& site, const std::string& index, const std::string& listed_sha256 = "", const std::string& location = "generated/index.json"){
        write_file(site / "version" / "1" / "generated" / "index.json", index);
        std::string sha256 = listed_sha256.empty() ? conta::detail::sha256_text(index) : listed_sha256;
        write_file(site / "version" / "1" / "meta.json", "{\"format\": 1, \"index\": {\"location\": \"" + location + "\", \"sha256\": \"" + sha256 + "\"}}\n");
        return file_url(site / "version" / "1" / "meta.json");
    }
    conta::Config index_config(const std::filesystem::path& scratch, const std::vector<std::string>& entry_points){
        conta::Config config;
        config.cache = (scratch / "cache").string();
        config.entry_points = entry_points;
        return config;
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

TEST(UTILS_CONTA, SHA256_KNOWN_VECTORS){
    ASSERT_EQ(conta::detail::sha256_text(""), EMPTY_SHA256);
    ASSERT_EQ(conta::detail::sha256_text("abc"), ABC_SHA256);
    ASSERT_EQ(conta::detail::sha256_text("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"), "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    ASSERT_EQ(conta::detail::sha256_text(std::string(1000000, 'a')), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    std::filesystem::path scratch = scratch_directory();
    write_file(scratch / "hello", "hello");
    std::string sha1, sha256;
    ASSERT_TRUE(conta::detail::digests_file(scratch / "hello", sha1, sha256));
    ASSERT_EQ(sha1, HELLO_SHA1);
    ASSERT_EQ(sha256, HELLO_SHA256);
}

TEST(UTILS_CONTA, REFERENCES){
    std::filesystem::path scratch = scratch_directory();
    conta::Config config;
    config.root = (scratch / "root").string();
    write_file(scratch / "root" / "data" / ABC_SHA1, "abc");
    std::string path, error;
    ASSERT_TRUE(conta::resolve(config, std::string("conta:") + ABC_SHA1, path, error)) << error;
    ASSERT_EQ(path, (scratch / "root" / "data" / ABC_SHA1).string());
    ASSERT_TRUE(conta::resolve(config, "A9993E364706816ABA3E25717850C26C9CD0D89D", path, error)) << error;
    ASSERT_FALSE(conta::resolve(config, std::string("conta:") + ABC_SHA256, path, error));
    ASSERT_NE(error.find("is a sha256"), std::string::npos);
}

TEST(UTILS_CONTA, INDEX_JSON_TOLERANCE){
    std::string text = std::string(R"({"future": {"nested": [1, 2.5e3, -0.5, true, false, null, {"deep": []}]},
"shards": {"a": {"type": "future-type", "url": "https://a.example/{sha1}", "license": {"spdx": "CC0-1.0"}}, "no-url": {"type": "torrent"}},
"entries": [
{"sha1": ")") + ABC_SHA1 + R"(", "sha256": ")" + ABC_SHA256 + R"(", "shards": ["unknown", 7, "a"], "description": "café 😀", "size": 3, "license": ["x"]},
{"sha1": ")" + ABC_SHA1 + R"(", "sha256": ")" + EMPTY_SHA256 + R"(", "shards": ["a"], "description": "duplicate"},
{"sha1": ")" + EMPTY_SHA1 + R"(", "sha256": "not-hex", "shards": ["a"]},
"not an object",
{"sha1": ")" + EMPTY_SHA1 + R"(", "sha256": ")" + EMPTY_SHA256 + R"(", "shards": "wrong type", "description": 5}
]})";
    conta::detail::Index index;
    ASSERT_TRUE(conta::detail::parse_index(text, {ABC_SHA1, EMPTY_SHA1}, index));
    ASSERT_EQ(index.entry_count, 3u);
    ASSERT_EQ(index.shard_urls.size(), 1u);
    ASSERT_EQ(index.shard_urls.at("a"), "https://a.example/{sha1}");
    const conta::detail::Entry& abc = index.entries.at(ABC_SHA1);
    ASSERT_EQ(abc.sha256, ABC_SHA256);
    ASSERT_EQ(abc.shards, (std::vector<std::string>{"unknown", "a"}));
    ASSERT_EQ(abc.description, "caf\xc3\xa9 \xf0\x9f\x98\x80");
    const conta::detail::Entry& empty = index.entries.at(EMPTY_SHA1);
    ASSERT_EQ(empty.sha256, EMPTY_SHA256);
    ASSERT_TRUE(empty.shards.empty());
    ASSERT_TRUE(empty.description.empty());
    ASSERT_FALSE(conta::detail::parse_index(R"({"shards": {}})", {}, index));
    ASSERT_FALSE(conta::detail::parse_index(text + " trailing", {}, index));

    std::string meta_text = std::string(R"({"format": 1, "index": {"location": "generated/index.json", "sha256": ")") + ABC_SHA256 + R"(", "size": 1}, "future": {"x": [1]}})";
    conta::detail::Meta meta;
    ASSERT_TRUE(conta::detail::parse_meta(meta_text, meta));
    ASSERT_TRUE(meta.format_supported);
    ASSERT_EQ(meta.location, "generated/index.json");
    ASSERT_EQ(meta.sha256, ABC_SHA256);
    for(const char* format: {"2", "1.0", "\"1\"", "true"}){
        conta::detail::Meta other;
        ASSERT_TRUE(conta::detail::parse_meta(std::string("{\"format\": ") + format + "}", other));
        ASSERT_FALSE(other.format_supported) << format;
    }
    ASSERT_TRUE(conta::detail::parse_meta("{\"format\": 1, \"x\": " + std::string(50, '[') + std::string(50, ']') + "}", meta));
    ASSERT_FALSE(conta::detail::parse_meta("{\"format\": 1, \"x\": " + std::string(100, '[') + std::string(100, ']') + "}", meta));
    ASSERT_FALSE(conta::detail::parse_meta("{\"format\": 1, \"x\": \"raw\ncontrol\"}", meta));
}

TEST(UTILS_CONTA, INDEX_RESOLVE_AND_CACHE){
    std::filesystem::path scratch = scratch_directory();
    write_file(scratch / "store" / ABC_SHA1, "abc");
    std::string index = index_text(local_shard(scratch / "store"), entry(ABC_SHA1, ABC_SHA256));
    conta::Config config = index_config(scratch, {publish(scratch / "site", index)});
    std::string path, error;
    ASSERT_TRUE(conta::resolve(config, ABC_SHA1, path, error)) << error;
    ASSERT_EQ(path, (scratch / "cache" / ABC_SHA1).string());
    ASSERT_EQ(read_text(scratch / "cache" / conta::CACHED_INDEX_NAME), index);
    std::filesystem::remove(scratch / "cache" / ABC_SHA1);
    config.entry_points = {file_url(scratch / "missing" / "meta.json")};
    ASSERT_TRUE(conta::resolve(config, ABC_SHA1, path, error)) << error;
}

TEST(UTILS_CONTA, ENTRY_POINT_FALLBACK){
    std::filesystem::path scratch = scratch_directory();
    write_file(scratch / "store" / ABC_SHA1, "abc");
    write_file(scratch / "future" / "meta.json", "{\"format\": 2, \"index\": {\"location\": \"index.json\", \"sha256\": \"" + std::string(EMPTY_SHA256) + "\"}}");
    std::string good = publish(scratch / "site", index_text(local_shard(scratch / "store"), entry(ABC_SHA1, ABC_SHA256)));
    conta::Config config = index_config(scratch, {file_url(scratch / "missing" / "meta.json"), file_url(scratch / "future" / "meta.json"), good});
    std::string path, error;
    ASSERT_TRUE(conta::resolve(config, ABC_SHA1, path, error)) << error;
}

TEST(UTILS_CONTA, INDEX_SHA256_MISMATCH_IS_NOT_CACHED){
    std::filesystem::path scratch = scratch_directory();
    write_file(scratch / "store" / ABC_SHA1, "abc");
    std::string index = index_text(local_shard(scratch / "store"), entry(ABC_SHA1, ABC_SHA256));
    conta::Config config = index_config(scratch, {publish(scratch / "site", index, EMPTY_SHA256)});
    std::string path, error;
    ASSERT_TRUE(conta::resolve(config, ABC_SHA1, path, error)) << error;
    ASSERT_FALSE(std::filesystem::exists(scratch / "cache" / conta::CACHED_INDEX_NAME));
}

TEST(UTILS_CONTA, ENTRY_SHA256_IS_VERIFIED){
    std::filesystem::path scratch = scratch_directory();
    write_file(scratch / "store" / ABC_SHA1, "abc");
    std::string index = index_text(local_shard(scratch / "store"), entry(ABC_SHA1, EMPTY_SHA256));
    conta::Config config = index_config(scratch, {publish(scratch / "site", index)});
    std::string path, error;
    ASSERT_FALSE(conta::resolve(config, ABC_SHA1, path, error));
    ASSERT_NE(error.find("sha256 mismatch"), std::string::npos) << error;
    ASSERT_FALSE(std::filesystem::exists(scratch / "cache" / ABC_SHA1));
}

TEST(UTILS_CONTA, UNINDEXED_REPORT){
    std::filesystem::path scratch = scratch_directory();
    write_file(scratch / "store" / ABC_SHA1, "abc");
    std::string meta_url = publish(scratch / "site", index_text(local_shard(scratch / "store"), ""));
    conta::Config config = index_config(scratch, {meta_url});
    std::string path, error;
    ASSERT_FALSE(conta::resolve(config, ABC_SHA1, path, error));
    ASSERT_NE(error.find("no reachable index lists"), std::string::npos) << error;
    ASSERT_NE(error.find(file_url(scratch / "site" / "version" / "1" / "generated" / "index.json") + ": does not list it"), std::string::npos) << error;
}

TEST(UTILS_CONTA, OLDER_INDEX_DOES_NOT_REPLACE_CACHE){
    std::filesystem::path scratch = scratch_directory();
    write_file(scratch / "store" / ABC_SHA1, "abc");
    std::string unreachable_shard = "\"local\": {\"type\": \"http\", \"url\": \"http://127.0.0.1:1/{sha1}\"}";
    std::string cached = index_text(unreachable_shard, entry(ABC_SHA1, ABC_SHA256) + ",\n" + entry(HELLO_SHA1, HELLO_SHA256));
    write_file(scratch / "cache" / conta::CACHED_INDEX_NAME, cached);
    std::string older = index_text(local_shard(scratch / "store"), entry(ABC_SHA1, ABC_SHA256));
    conta::Config config = index_config(scratch, {publish(scratch / "older", older)});
    std::string path, error;
    ASSERT_TRUE(conta::resolve(config, ABC_SHA1, path, error)) << error;
    ASSERT_EQ(read_text(scratch / "cache" / conta::CACHED_INDEX_NAME), cached);
    std::filesystem::remove(scratch / "cache" / ABC_SHA1);
    std::string newer = index_text(local_shard(scratch / "store"), entry(ABC_SHA1, ABC_SHA256) + ",\n" + entry(HELLO_SHA1, HELLO_SHA256) + ",\n" + entry(EMPTY_SHA1, EMPTY_SHA256));
    config.entry_points = {publish(scratch / "newer", newer)};
    ASSERT_TRUE(conta::resolve(config, ABC_SHA1, path, error)) << error;
    ASSERT_EQ(read_text(scratch / "cache" / conta::CACHED_INDEX_NAME), newer);
}

TEST(UTILS_CONTA, UNSAFE_LOCATIONS_ARE_SKIPPED){
    std::filesystem::path scratch = scratch_directory();
    std::string shards = R"("unc": {"url": "file:////server/share/{sha1}"},
"percent": {"url": "https://example.invalid/%HOME%/{sha1}"},
"placeholder": {"url": "https://example.invalid/{sha1}/{size}"},
"quote": {"url": "https://example.invalid/'{sha1}"})";
    std::string index = index_text(shards, entry(ABC_SHA1, ABC_SHA256, R"("unc", "percent", "placeholder", "quote")"));
    conta::Config config = index_config(scratch, {publish(scratch / "site", index)});
    std::string path, error;
    ASSERT_FALSE(conta::resolve(config, ABC_SHA1, path, error));
    for(const char* shard: {"unc", "percent", "placeholder", "quote"}){
        ASSERT_NE(error.find(std::string("skipped shard ") + shard), std::string::npos) << error;
    }
    ASSERT_EQ(error.find("curl exit"), std::string::npos) << error;
}

TEST(UTILS_CONTA, UNSUPPORTED_INDEX_LOCATION){
    std::filesystem::path scratch = scratch_directory();
    write_file(scratch / "store" / ABC_SHA1, "abc");
    std::string index = index_text(local_shard(scratch / "store"), entry(ABC_SHA1, ABC_SHA256));
    conta::Config config = index_config(scratch, {publish(scratch / "site", index, "", "../escape/index.json")});
    std::string path, error;
    ASSERT_FALSE(conta::resolve(config, ABC_SHA1, path, error));
    ASSERT_NE(error.find("unsupported index.location"), std::string::npos) << error;
}

TEST(UTILS_CONTA, BATCH_THROUGH_INDEX){
    std::filesystem::path scratch = scratch_directory();
    write_file(scratch / "store" / ABC_SHA1, "abc");
    write_file(scratch / "store" / HELLO_SHA1, "hello");
    std::string index = index_text(local_shard(scratch / "store"), entry(ABC_SHA1, ABC_SHA256) + ",\n" + entry(HELLO_SHA1, HELLO_SHA256));
    conta::Config config = index_config(scratch, {publish(scratch / "site", index)});
    std::vector<std::string> paths;
    std::string error;
    ASSERT_TRUE(conta::resolve(config, std::vector<std::string>{ABC_SHA1, std::string("conta:") + HELLO_SHA1}, paths, error)) << error;
    ASSERT_EQ(paths, (std::vector<std::string>{(scratch / "cache" / ABC_SHA1).string(), (scratch / "cache" / HELLO_SHA1).string()}));
}

TEST(UTILS_CONTA, NETWORK_INDEX){
    const char* enabled = std::getenv("RL_TOOLS_TEST_CONTA_NETWORK");
    if(enabled == nullptr || std::string(enabled) != "1"){
        GTEST_SKIP() << "set RL_TOOLS_TEST_CONTA_NETWORK=1 to enable";
    }
    std::filesystem::path scratch = scratch_directory();
    conta::Config config;
    config.cache = (scratch / "cache").string();
    constexpr char BLOB[] = "b6bd082c60fcb27f42a2b5f2f635587d2b60d170";
    std::string path, error;
    ASSERT_TRUE(conta::resolve(config, BLOB, path, error)) << error;
    std::string hex;
    ASSERT_TRUE(conta::detail::sha1_file(path, hex));
    ASSERT_EQ(hex, BLOB);
    ASSERT_TRUE(std::filesystem::exists(scratch / "cache" / conta::CACHED_INDEX_NAME));
}

TEST(UTILS_CONTA, NETWORK_DOWNLOAD){
    const char* enabled = std::getenv("RL_TOOLS_TEST_CONTA_NETWORK");
    if(enabled == nullptr || std::string(enabled) != "1"){
        GTEST_SKIP() << "set RL_TOOLS_TEST_CONTA_NETWORK=1 to enable";
    }
    std::filesystem::path scratch = scratch_directory();
    conta::Config config;
    config.cache = (scratch / "cache").string();
    config.url_base = "https://huggingface.co/datasets/rl-tools/conta/resolve/main/data/";
    constexpr char BLOB[] = "a38b2994e7674f467fe81e86d2f21c45bfd965a0";
    std::string path, error;
    ASSERT_TRUE(conta::resolve(config, BLOB, path, error)) << error;
    std::string hex;
    ASSERT_TRUE(conta::detail::sha1_file(path, hex));
    ASSERT_EQ(hex, BLOB);
}

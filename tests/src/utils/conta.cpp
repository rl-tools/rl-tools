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
        std::string generic = path.generic_string();
        return generic.front() == '/' ? "file://" + generic : "file:///" + generic;
    }
    std::string entry(const std::string& sha1, const std::string& sha256, const std::string& shards = R"({"id": "local", "compressed": []})"){
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
    std::string gzip_abc(){
        return std::string("\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\x03\x4b\x4c\x4a\x06\x00\xc2\x41\x24\x35\x03\x00\x00\x00", 23);
    }
    std::string gzip_entry(const std::string& sha256 = ABC_SHA256){
        return entry(ABC_SHA1, sha256, R"({"id": "local", "compressed": ["future", "gz"]})");
    }
    void expect_no_partials(const std::filesystem::path& directory){
        for(const auto& item: std::filesystem::directory_iterator(directory)){
            EXPECT_EQ(item.path().filename().string().find(".partial."), std::string::npos) << item.path();
        }
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
    config.url_base = file_url(scratch / "store") + "/";
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
    config.url_base = file_url(scratch / "store") + "/";
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
    config.url_base = file_url(scratch / "store") + "/";
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
{"sha1": ")") + ABC_SHA1 + R"(", "sha256": ")" + ABC_SHA256 + R"(", "shards": [{"id": "unknown"}, 7, "old-string", {"id": "a", "compressed": ["future", 1, "gz"], "future": true}], "description": "café 😀", "size": 3, "license": ["x"]},
{"sha1": ")" + ABC_SHA1 + R"(", "sha256": ")" + EMPTY_SHA256 + R"(", "shards": [{"id": "a"}], "description": "duplicate"},
{"sha1": ")" + EMPTY_SHA1 + R"(", "sha256": "not-hex", "shards": [{"id": "a"}]},
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
    ASSERT_EQ(abc.shards.size(), 2u);
    ASSERT_EQ(abc.shards[0].id, "unknown");
    ASSERT_TRUE(abc.shards[0].compressed.empty());
    ASSERT_EQ(abc.shards[1].id, "a");
    ASSERT_EQ(abc.shards[1].compressed, (std::vector<std::string>{"future", "gz"}));
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
    std::string unreachable_shard = local_shard(scratch / "missing-store");
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
    std::string index = index_text(shards, entry(ABC_SHA1, ABC_SHA256, R"({"id": "unc"}, {"id": "percent"}, {"id": "placeholder"}, {"id": "quote"})"));
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

TEST(UTILS_CONTA, DEFAULT_CACHE_DIRECTORY){
    std::filesystem::path scratch = scratch_directory();
    const char* names[] = {"CONTA_CACHE", "XDG_CACHE_HOME", "HOME", "LOCALAPPDATA"};
    std::vector<std::pair<std::string, bool>> saved;
    for(const char* name: names){
        const char* value = std::getenv(name);
        saved.emplace_back(value != nullptr ? value : "", value != nullptr);
    }
    auto set = [](const char* name, const std::string& value){
#if defined(_WIN32)
        _putenv_s(name, value.c_str());
#else
        if(value.empty()){
            unsetenv(name);
        }
        else{
            setenv(name, value.c_str(), 1);
        }
#endif
    };
    set("CONTA_CACHE", "");
    set("XDG_CACHE_HOME", (scratch / "xdg").string());
    set("HOME", scratch.string());
    set("LOCALAPPDATA", (scratch / "local").string());
#if defined(_WIN32)
    EXPECT_EQ(conta::config_from_environment().cache, (scratch / "local" / "conta").string());
#else
    EXPECT_EQ(conta::config_from_environment().cache, (scratch / "xdg" / "conta").string());
    set("XDG_CACHE_HOME", "");
    EXPECT_EQ(conta::config_from_environment().cache, (scratch / ".cache" / "conta").string());
#endif
    set("CONTA_CACHE", (scratch / "explicit").string());
    EXPECT_EQ(conta::config_from_environment().cache, (scratch / "explicit").string());
    for(std::size_t name_i = 0; name_i < saved.size(); name_i++){
        set(names[name_i], saved[name_i].second ? saved[name_i].first : std::string());
    }
}

TEST(UTILS_CONTA, LOAD_PLAIN_CACHE_AND_ROOT){
    std::filesystem::path scratch = scratch_directory();
    conta::Config config = index_config(scratch, {});
    write_file(scratch / "cache" / ABC_SHA1, "abc");
    write_file(scratch / "cache" / (std::string(ABC_SHA1) + ".gz"), "invalid gzip is never consulted");
    std::vector<std::uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(conta::load(config, std::string("conta:") + ABC_SHA1, bytes, error)) << error;
    ASSERT_EQ(bytes, (std::vector<std::uint8_t>{'a', 'b', 'c'}));
    config.root = (scratch / "root").string();
    write_file(scratch / "root" / "data" / ABC_SHA1, "abc");
    config.cache.clear();
    ASSERT_TRUE(conta::load(config, ABC_SHA1, bytes, error)) << error;
    ASSERT_EQ(bytes, (std::vector<std::uint8_t>{'a', 'b', 'c'}));
    ASSERT_FALSE(conta::load(config, "invalid", bytes, error));
    ASSERT_TRUE(bytes.empty());
}

TEST(UTILS_CONTA, LOAD_RAW_SOURCE){
    std::filesystem::path scratch = scratch_directory();
    write_file(scratch / "store" / ABC_SHA1, "abc");
    conta::Config config = index_config(scratch, {publish(scratch / "site", index_text(local_shard(scratch / "store"), entry(ABC_SHA1, ABC_SHA256, R"({"id": "local", "compressed": ["future"]})")))});
    std::vector<std::uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(conta::load(config, ABC_SHA1, bytes, error)) << error;
    ASSERT_EQ(bytes, (std::vector<std::uint8_t>{'a', 'b', 'c'}));
    ASSERT_EQ(read_text(scratch / "cache" / ABC_SHA1), "abc");
    ASSERT_FALSE(std::filesystem::exists(scratch / "cache" / (std::string(ABC_SHA1) + ".gz")));
}

TEST(UTILS_CONTA, COMPRESSED_CACHE_SETTING){
    const char* name = "CONTA_COMPRESSED_CACHE";
    const char* original = std::getenv(name);
    bool was_set = original != nullptr;
    std::string saved = was_set ? original : "";
    auto set = [&](const std::string& value){
#if defined(_WIN32)
        _putenv_s(name, value.c_str());
#else
        if(value.empty()){
            unsetenv(name);
        }
        else{
            setenv(name, value.c_str(), 1);
        }
#endif
    };
    set("");
    EXPECT_TRUE(conta::config_from_environment().compressed_cache);
    set("0");
    EXPECT_FALSE(conta::config_from_environment().compressed_cache);
    EXPECT_TRUE(conta::Config{}.compressed_cache);
    set("1");
    EXPECT_TRUE(conta::config_from_environment().compressed_cache);
    set("unsupported");
    EXPECT_TRUE(conta::config_from_environment().compressed_cache);
    set(was_set ? saved : "");
}

TEST(UTILS_CONTA, GZIP_URL_EXTENSION){
    ASSERT_EQ(conta::detail::compressed_location("https://example.com/data/hash?download=true"), "https://example.com/data/hash.gz?download=true");
    ASSERT_EQ(conta::detail::compressed_location("file:///data/hash"), "file:///data/hash.gz");
    ASSERT_EQ(conta::detail::compressed_location("https://example.com/data/hash?download=true#asset"), "https://example.com/data/hash.gz?download=true#asset");
    ASSERT_EQ(conta::detail::compressed_location("https://example.com/data/hash#asset"), "https://example.com/data/hash.gz#asset");
    ASSERT_TRUE(conta::detail::is_document_url("https://example.com/data/hash?download=true#asset"));
}

TEST(UTILS_CONTA, CACHE_PUBLICATION){
    std::filesystem::path scratch = scratch_directory();
    write_file(scratch / "target", std::string(100000, 'a'));
    write_file(scratch / "temporary", std::string(100000, 'a'));
    ASSERT_TRUE(conta::detail::same_file_contents(scratch / "temporary", scratch / "target"));
    std::string error;
    ASSERT_TRUE(conta::detail::publish_file(scratch / "temporary", scratch / "target", error)) << error;
    ASSERT_FALSE(std::filesystem::exists(scratch / "temporary"));
    ASSERT_EQ(read_text(scratch / "target"), std::string(100000, 'a'));
    write_file(scratch / "temporary", std::string(99999, 'a') + "b");
    ASSERT_FALSE(conta::detail::same_file_contents(scratch / "temporary", scratch / "target"));
    write_file(scratch / "temporary", "shorter");
    ASSERT_FALSE(conta::detail::same_file_contents(scratch / "temporary", scratch / "target"));
    ASSERT_FALSE(conta::detail::same_file_contents(scratch / "missing", scratch / "target"));
    std::filesystem::create_directory(scratch / "directory");
    ASSERT_FALSE(conta::detail::publish_file(scratch / "temporary", scratch / "directory", error));
    ASSERT_FALSE(std::filesystem::exists(scratch / "temporary"));
    ASSERT_TRUE(std::filesystem::is_directory(scratch / "directory"));
}

#if defined(RL_TOOLS_ENABLE_ZLIB) || defined(CONTA_ENABLE_ZLIB)
TEST(UTILS_CONTA, LOAD_GZIP_CACHE_AND_MATERIALIZE){
    std::filesystem::path scratch = scratch_directory();
    conta::Config config = index_config(scratch, {});
    std::filesystem::path gzip_path = scratch / "cache" / (std::string(ABC_SHA1) + ".gz");
    write_file(gzip_path, gzip_abc());
    std::vector<std::uint8_t> bytes;
    std::string path, error;
    ASSERT_TRUE(conta::load(config, ABC_SHA1, bytes, error)) << error;
    ASSERT_EQ(bytes, (std::vector<std::uint8_t>{'a', 'b', 'c'}));
    ASSERT_FALSE(std::filesystem::exists(scratch / "cache" / ABC_SHA1));
    ASSERT_EQ(read_text(gzip_path), gzip_abc());
    ASSERT_TRUE(conta::resolve(config, ABC_SHA1, path, error)) << error;
    ASSERT_EQ(path, (scratch / "cache" / ABC_SHA1).string());
    ASSERT_EQ(read_text(path), "abc");
    ASSERT_EQ(read_text(gzip_path), gzip_abc());
    write_file(gzip_path, "corrupt gzip must be ignored while plain is cached");
    ASSERT_TRUE(conta::load(config, ABC_SHA1, bytes, error)) << error;
    ASSERT_EQ(bytes, (std::vector<std::uint8_t>{'a', 'b', 'c'}));
    expect_no_partials(scratch / "cache");
}

TEST(UTILS_CONTA, LOAD_PLAIN_CACHE_POLICY){
    std::filesystem::path scratch = scratch_directory();
    conta::Config config = index_config(scratch, {});
    config.compressed_cache = false;
    std::filesystem::path gzip_path = scratch / "cache" / (std::string(ABC_SHA1) + ".gz");
    write_file(gzip_path, gzip_abc());
    std::vector<std::uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(conta::load(config, ABC_SHA1, bytes, error)) << error;
    ASSERT_EQ(bytes, (std::vector<std::uint8_t>{'a', 'b', 'c'}));
    ASSERT_EQ(read_text(scratch / "cache" / ABC_SHA1), "abc");
    auto modified = std::filesystem::last_write_time(scratch / "cache" / ABC_SHA1);
    write_file(gzip_path, "subsequent loads must use plain");
    ASSERT_TRUE(conta::load(config, ABC_SHA1, bytes, error)) << error;
    ASSERT_EQ(std::filesystem::last_write_time(scratch / "cache" / ABC_SHA1), modified);
    expect_no_partials(scratch / "cache");
}

TEST(UTILS_CONTA, ROOT_GZIP_MEMORY_AND_PATH){
    std::filesystem::path scratch = scratch_directory();
    conta::Config config;
    config.root = (scratch / "root").string();
    std::filesystem::path gzip_path = scratch / "root" / "data" / (std::string(ABC_SHA1) + ".gz");
    write_file(gzip_path, gzip_abc());
    std::vector<std::uint8_t> bytes;
    std::string path, error;
    ASSERT_TRUE(conta::load(config, ABC_SHA1, bytes, error)) << error;
    ASSERT_EQ(bytes, (std::vector<std::uint8_t>{'a', 'b', 'c'}));
    ASSERT_FALSE(conta::resolve(config, ABC_SHA1, path, error));
    ASSERT_NE(error.find("CONTA_CACHE"), std::string::npos);
    config.cache = (scratch / "cache").string();
    ASSERT_TRUE(conta::resolve(config, ABC_SHA1, path, error)) << error;
    ASSERT_EQ(path, (scratch / "cache" / ABC_SHA1).string());
    ASSERT_EQ(read_text(path), "abc");
    ASSERT_FALSE(std::filesystem::exists(scratch / "root" / "data" / ABC_SHA1));
    ASSERT_EQ(read_text(gzip_path), gzip_abc());
    auto modified = std::filesystem::last_write_time(path);
    ASSERT_TRUE(conta::resolve(config, ABC_SHA1, path, error)) << error;
    ASSERT_EQ(std::filesystem::last_write_time(path), modified);
    std::filesystem::remove(gzip_path);
    ASSERT_FALSE(conta::resolve(config, ABC_SHA1, path, error));
    ASSERT_NE(error.find("CONTA_ROOT"), std::string::npos);
}

TEST(UTILS_CONTA, ROOT_GZIP_LFS_POINTER){
    std::filesystem::path scratch = scratch_directory();
    conta::Config config = index_config(scratch, {});
    config.root = (scratch / "root").string();
    write_file(scratch / "root" / "data" / (std::string(ABC_SHA1) + ".gz"), "version https://git-lfs.github.com/spec/v1\noid sha256:0000000000000000000000000000000000000000000000000000000000000000\nsize 23\n");
    std::vector<std::uint8_t> bytes;
    std::string error;
    ASSERT_FALSE(conta::load(config, ABC_SHA1, bytes, error));
    ASSERT_NE(error.find("git lfs pull"), std::string::npos);
    ASSERT_TRUE(bytes.empty());
    write_file(scratch / "cache" / ABC_SHA1, "abc");
    ASSERT_FALSE(conta::load(config, ABC_SHA1, bytes, error));
    ASSERT_NE(error.find("git lfs pull"), std::string::npos);
}

TEST(UTILS_CONTA, DOWNLOAD_GZIP_RETAINED_FOR_LOAD){
    std::filesystem::path scratch = scratch_directory();
    write_file(scratch / "store" / (std::string(ABC_SHA1) + ".gz"), gzip_abc());
    conta::Config config = index_config(scratch, {publish(scratch / "site", index_text(local_shard(scratch / "store"), gzip_entry()))});
    std::vector<std::uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(conta::load(config, ABC_SHA1, bytes, error)) << error;
    ASSERT_EQ(bytes, (std::vector<std::uint8_t>{'a', 'b', 'c'}));
    ASSERT_FALSE(std::filesystem::exists(scratch / "cache" / ABC_SHA1));
    ASSERT_EQ(read_text(scratch / "cache" / (std::string(ABC_SHA1) + ".gz")), gzip_abc());
    config.entry_points.clear();
    std::filesystem::remove_all(scratch / "store");
    ASSERT_TRUE(conta::load(config, ABC_SHA1, bytes, error)) << error;
    expect_no_partials(scratch / "cache");
}

TEST(UTILS_CONTA, DOWNLOAD_GZIP_RETAINED_PLAIN_FOR_RESOLVE){
    std::filesystem::path scratch = scratch_directory();
    write_file(scratch / "store" / (std::string(ABC_SHA1) + ".gz"), gzip_abc());
    std::string meta = publish(scratch / "site", index_text(local_shard(scratch / "store"), gzip_entry()));
    conta::Config config = index_config(scratch, {meta});
    std::string path, error;
    ASSERT_TRUE(conta::resolve(config, ABC_SHA1, path, error)) << error;
    ASSERT_EQ(read_text(path), "abc");
    ASSERT_FALSE(std::filesystem::exists(scratch / "cache" / (std::string(ABC_SHA1) + ".gz")));
    config.cache = (scratch / "plain-load-cache").string();
    config.compressed_cache = false;
    std::vector<std::uint8_t> bytes;
    ASSERT_TRUE(conta::load(config, ABC_SHA1, bytes, error)) << error;
    ASSERT_EQ(bytes, (std::vector<std::uint8_t>{'a', 'b', 'c'}));
    ASSERT_EQ(read_text(scratch / "plain-load-cache" / ABC_SHA1), "abc");
    ASSERT_FALSE(std::filesystem::exists(scratch / "plain-load-cache" / (std::string(ABC_SHA1) + ".gz")));
    expect_no_partials(scratch / "cache");
    expect_no_partials(scratch / "plain-load-cache");
}

TEST(UTILS_CONTA, CORRUPT_GZIP_FALLS_BACK_TO_RAW){
    std::filesystem::path scratch = scratch_directory();
    write_file(scratch / "store" / (std::string(ABC_SHA1) + ".gz"), gzip_abc().substr(0, 19));
    write_file(scratch / "store" / ABC_SHA1, "abc");
    write_file(scratch / "cache" / (std::string(ABC_SHA1) + ".gz"), "corrupt cached gzip");
    conta::Config config = index_config(scratch, {publish(scratch / "site", index_text(local_shard(scratch / "store"), gzip_entry()))});
    std::vector<std::uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(conta::load(config, ABC_SHA1, bytes, error)) << error;
    ASSERT_EQ(bytes, (std::vector<std::uint8_t>{'a', 'b', 'c'}));
    ASSERT_EQ(read_text(scratch / "cache" / ABC_SHA1), "abc");
    ASSERT_EQ(read_text(scratch / "cache" / (std::string(ABC_SHA1) + ".gz")), "corrupt cached gzip");
    expect_no_partials(scratch / "cache");
}

TEST(UTILS_CONTA, GZIP_VERIFIES_ORIGINAL_HASHES){
    std::filesystem::path scratch = scratch_directory();
    write_file(scratch / "store" / (std::string(ABC_SHA1) + ".gz"), gzip_abc());
    conta::Config config = index_config(scratch, {publish(scratch / "site", index_text(local_shard(scratch / "store"), gzip_entry(EMPTY_SHA256)))});
    std::vector<std::uint8_t> bytes{'x'};
    std::string error;
    ASSERT_FALSE(conta::load(config, ABC_SHA1, bytes, error));
    ASSERT_NE(error.find("sha256 mismatch"), std::string::npos) << error;
    ASSERT_TRUE(bytes.empty());
    ASSERT_FALSE(std::filesystem::exists(scratch / "cache" / ABC_SHA1));
    ASSERT_FALSE(std::filesystem::exists(scratch / "cache" / (std::string(ABC_SHA1) + ".gz")));
    expect_no_partials(scratch / "cache");
    config.entry_points.clear();
    write_file(scratch / "cache" / (std::string(ABC_SHA1) + ".gz"), gzip_abc());
    ASSERT_FALSE(conta::load(config, ABC_SHA1, bytes, error));
    ASSERT_NE(error.find("sha256 mismatch"), std::string::npos) << error;
    config.root = (scratch / "root").string();
    write_file(scratch / "root" / "data" / (std::string(EMPTY_SHA1) + ".gz"), gzip_abc());
    ASSERT_FALSE(conta::load(config, EMPTY_SHA1, bytes, error));
    ASSERT_NE(error.find("hash mismatch"), std::string::npos) << error;
    ASSERT_TRUE(bytes.empty());
}

TEST(UTILS_CONTA, GZIP_STREAM_VALIDATION){
    std::filesystem::path scratch = scratch_directory();
    conta::Config config = index_config(scratch, {});
    config.root = (scratch / "root").string();
    std::filesystem::path source = scratch / "root" / "data" / (std::string(ABC_SHA1) + ".gz");
    std::vector<std::string> invalid = {"abc", gzip_abc().substr(0, 19), gzip_abc() + "garbage"};
    std::string bad_crc = gzip_abc();
    bad_crc[15] = 0;
    invalid.push_back(bad_crc);
    for(const std::string& content: invalid){
        write_file(source, content);
        std::vector<std::uint8_t> bytes{'x'};
        std::string path, error;
        ASSERT_FALSE(conta::load(config, ABC_SHA1, bytes, error));
        ASSERT_TRUE(bytes.empty());
        ASSERT_FALSE(conta::resolve(config, ABC_SHA1, path, error));
        ASSERT_FALSE(std::filesystem::exists(scratch / "cache" / ABC_SHA1));
        expect_no_partials(scratch / "cache");
    }
    std::string double_abc = "abcabc";
    conta::detail::Sha1 sha1;
    sha1.update(reinterpret_cast<const unsigned char*>(double_abc.data()), double_abc.size());
    std::string hash = sha1.finalize();
    write_file(scratch / "root" / "data" / (hash + ".gz"), gzip_abc() + gzip_abc());
    std::vector<std::uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(conta::load(config, hash, bytes, error)) << error;
    ASSERT_EQ(std::string(bytes.begin(), bytes.end()), double_abc);
}

TEST(UTILS_CONTA, GZIP_LARGE_STREAM_AND_EMPTY){
    std::filesystem::path scratch = scratch_directory();
    conta::Config config = index_config(scratch, {});
    config.root = (scratch / "root").string();
    for(const std::string& content: {std::string(), std::string(200000, 'a')}){
        conta::detail::Sha1 sha1;
        sha1.update(reinterpret_cast<const unsigned char*>(content.data()), content.size());
        std::string hash = sha1.finalize();
        std::filesystem::path source = scratch / "root" / "data" / (hash + ".gz");
        std::filesystem::create_directories(source.parent_path());
        gzFile gzip = gzopen(source.string().c_str(), "wb9");
        ASSERT_NE(gzip, nullptr);
        int written = gzwrite(gzip, content.data(), unsigned(content.size()));
        int closed = gzclose(gzip);
        ASSERT_EQ(written, int(content.size()));
        ASSERT_EQ(closed, Z_OK);
        std::vector<std::uint8_t> bytes;
        std::string path, error;
        ASSERT_TRUE(conta::load(config, hash, bytes, error)) << error;
        ASSERT_EQ(std::string(bytes.begin(), bytes.end()), content);
        ASSERT_FALSE(std::filesystem::exists(scratch / "cache" / hash));
        ASSERT_TRUE(conta::resolve(config, hash, path, error)) << error;
        ASSERT_EQ(read_text(path), content);
    }
}
#else
TEST(UTILS_CONTA, GZIP_UNAVAILABLE_USES_RAW){
    std::filesystem::path scratch = scratch_directory();
    write_file(scratch / "store" / ABC_SHA1, "abc");
    conta::Config config = index_config(scratch, {publish(scratch / "site", index_text(local_shard(scratch / "store"), gzip_entry()))});
    std::vector<std::uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(conta::load(config, ABC_SHA1, bytes, error)) << error;
    ASSERT_EQ(bytes, (std::vector<std::uint8_t>{'a', 'b', 'c'}));
    ASSERT_EQ(read_text(scratch / "cache" / ABC_SHA1), "abc");
    expect_no_partials(scratch / "cache");
}

TEST(UTILS_CONTA, ROOT_GZIP_UNAVAILABLE){
    std::filesystem::path scratch = scratch_directory();
    conta::Config config = index_config(scratch, {});
    config.root = (scratch / "root").string();
    write_file(scratch / "root" / "data" / (std::string(ABC_SHA1) + ".gz"), gzip_abc());
    std::vector<std::uint8_t> bytes;
    std::string path, error;
    ASSERT_FALSE(conta::load(config, ABC_SHA1, bytes, error));
    ASSERT_NE(error.find("CONTA_ENABLE_ZLIB"), std::string::npos) << error;
    ASSERT_FALSE(conta::resolve(config, ABC_SHA1, path, error));
    ASSERT_NE(error.find("plain blob"), std::string::npos) << error;
    ASSERT_FALSE(std::filesystem::exists(scratch / "cache"));
}
#endif

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

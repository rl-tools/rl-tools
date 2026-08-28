#include <metra/metra.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

TEST(UTILS_METRA, JSON_ESCAPE){
    ASSERT_EQ(metra::detail::json_escape("plain"), "plain");
    ASSERT_EQ(metra::detail::json_escape("quote\"backslash\\"), "quote\\\"backslash\\\\");
    ASSERT_EQ(metra::detail::json_escape("line\nbreak\ttab"), "line\\u000abreak\\u0009tab");
    ASSERT_EQ(metra::detail::json_escape(std::string(1, '\x01')), "\\u0001");
    ASSERT_EQ(metra::detail::json_escape("unicode \xc3\xa9"), "unicode \xc3\xa9");
}

TEST(UTILS_METRA, JSON_NUMBER){
    ASSERT_EQ(metra::detail::json_number(1.5), "1.5");
    ASSERT_EQ(metra::detail::json_number(-2), "-2");
    ASSERT_EQ(std::stod(metra::detail::json_number(0.1)), 0.1);
    ASSERT_EQ(metra::detail::json_number(std::nan("")), "null");
    ASSERT_EQ(metra::detail::json_number(std::numeric_limits<double>::infinity()), "null");
    ASSERT_EQ(metra::detail::json_number(-std::numeric_limits<double>::infinity()), "null");
}

TEST(UTILS_METRA, JSON_ARRAY){
    ASSERT_EQ(metra::detail::json_array({}), "[]");
    ASSERT_EQ(metra::detail::json_array({1.5, 2.5}), "[1.5,2.5]");
}

TEST(UTILS_METRA, BUILD_PAYLOAD){
    metra::Config config;
    config.url = "http://localhost:13340";
    config.commit = "abc";
    config.run = "run";
    ASSERT_EQ(metra::detail::build_payload(config, "metric", "1.5"), "{\"commit\":\"abc\",\"run\":\"run\",\"name\":\"metric\",\"value\":1.5}");
    ASSERT_EQ(metra::detail::build_payload(config, "quo\"te", "[1]"), "{\"commit\":\"abc\",\"run\":\"run\",\"name\":\"quo\\\"te\",\"value\":[1]}");
}

TEST(UTILS_METRA, SHELL_QUOTE){
    std::string quoted;
#if defined(_WIN32)
    ASSERT_TRUE(metra::detail::shell_quote("C:\\path with spaces", quoted));
    ASSERT_EQ(quoted, "\"C:\\path with spaces\"");
    ASSERT_FALSE(metra::detail::shell_quote("has\"quote", quoted));
#else
    ASSERT_TRUE(metra::detail::shell_quote("/path with spaces", quoted));
    ASSERT_EQ(quoted, "'/path with spaces'");
    ASSERT_FALSE(metra::detail::shell_quote("has'quote", quoted));
#endif
}

TEST(UTILS_METRA, RUN_ID){
    std::string first = metra::detail::default_run_id();
    std::string second = metra::detail::default_run_id();
    ASSERT_FALSE(first.empty());
    ASSERT_NE(first, second);
    std::string quoted;
    ASSERT_TRUE(metra::detail::shell_quote(first, quoted));
}

#if !defined(_WIN32)
TEST(UTILS_METRA, CONFIG_FROM_ENVIRONMENT){
    setenv("METRA_URL", "http://localhost:13340/", 1);
    setenv("METRA_COMMIT", "0123456789abcdef0123456789abcdef01234567", 1);
    setenv("METRA_RUN", "custom-run", 1);
    metra::Config config = metra::config_from_environment();
    ASSERT_EQ(config.url, "http://localhost:13340");
    ASSERT_EQ(config.commit, "0123456789abcdef0123456789abcdef01234567");
    ASSERT_EQ(config.run, "custom-run");
    unsetenv("METRA_URL");
    unsetenv("METRA_COMMIT");
    unsetenv("METRA_RUN");
    config = metra::config_from_environment();
    ASSERT_TRUE(config.url.empty());
    ASSERT_FALSE(config.commit.empty());
    ASSERT_FALSE(config.run.empty());
}
#endif

TEST(UTILS_METRA, UNCONFIGURED_IS_NOOP){
    metra::Config config;
    config.commit = "abc";
    config.run = "run";
    std::string error;
    ASSERT_TRUE(metra::log(config, "metric", 1.0, error));
    ASSERT_TRUE(error.empty());
}

TEST(UTILS_METRA, EMPTY_NAME){
    metra::Config config;
    config.url = "http://localhost:13340";
    config.commit = "abc";
    config.run = "run";
    std::string error;
    ASSERT_FALSE(metra::log(config, "", 1.0, error));
    ASSERT_NE(error.find("name"), std::string::npos);
}

TEST(UTILS_METRA, NETWORK_LOG){
    const char* enabled = std::getenv("RL_TOOLS_TEST_METRA_NETWORK");
    const char* url = std::getenv("METRA_URL");
    if(enabled == nullptr || std::string(enabled) != "1" || url == nullptr){
        GTEST_SKIP() << "set RL_TOOLS_TEST_METRA_NETWORK=1 and METRA_URL to enable";
    }
    std::string error;
    ASSERT_TRUE(metra::log(metra::config_from_environment(), "metra_test/network", 1.0, error)) << error;
}

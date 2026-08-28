#ifndef METRA_METRA_H
#define METRA_METRA_H
// metra: one-liner metric logging (commit/run/name/value) to a metra server (tools/metra), e.g. metra::log("pendulum/return", 1.23);
// Freestanding, C++17, STL-only; POSTs via the curl CLI. Holds a static per-process run id (a deliberate exception to the rl_tools no-global-state convention).
// METRA_URL: server base URL, unset disables logging. METRA_COMMIT: overrides commit detection (git rev-parse HEAD). METRA_COMMIT_TIME: overrides commit time detection (git show -s --format=%ct HEAD, unix seconds). METRA_RUN: overrides the generated run id.
// The METRA_COMMIT/METRA_COMMIT_TIME compile definitions (set by cmake/autodetect/git-hash.cmake at configure time) are fallbacks for binaries that run outside a git checkout.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <random>
#include <system_error>
#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#define METRA_STRINGIFY_INNER(x) #x
#define METRA_STRINGIFY(x) METRA_STRINGIFY_INNER(x)

namespace metra{
    struct Config{
        std::string url;
        std::string commit;
        std::string commit_time;
        std::string run;
    };
    namespace detail{
        inline bool shell_quote(const std::string& input, std::string& output){
#if defined(_WIN32)
            if(input.find('"') != std::string::npos || input.find('%') != std::string::npos){
                return false;
            }
            output = "\"" + input + "\"";
#else
            if(input.find('\'') != std::string::npos){
                return false;
            }
            output = "'" + input + "'";
#endif
            return true;
        }
        inline long process_id(){
#if defined(_WIN32)
            return _getpid();
#else
            return long(getpid());
#endif
        }
        inline std::string json_escape(const std::string& input){
            std::string output;
            output.reserve(input.size());
            for(char character: input){
                unsigned char code = (unsigned char)character;
                if(character == '"'){
                    output += "\\\"";
                }
                else if(character == '\\'){
                    output += "\\\\";
                }
                else if(code < 0x20){
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", code);
                    output += buffer;
                }
                else{
                    output += character;
                }
            }
            return output;
        }
        inline std::string json_number(double value){
            if(!std::isfinite(value)){ // JSON has no NaN/Inf; a null row still records that the run produced a non-finite value
                return "null";
            }
            char buffer[32];
            std::snprintf(buffer, sizeof(buffer), "%.17g", value);
            return buffer;
        }
        inline std::string json_array(const std::vector<double>& values){
            std::string output = "[";
            for(std::size_t value_i = 0; value_i < values.size(); value_i++){
                if(value_i > 0){
                    output += ",";
                }
                output += json_number(values[value_i]);
            }
            output += "]";
            return output;
        }
        inline std::string random_hex(unsigned length){
            std::random_device device;
            std::string hex;
            hex.reserve(length);
            for(unsigned character_i = 0; character_i < length; character_i++){
                hex += "0123456789abcdef"[device() & 0xFu];
            }
            return hex;
        }
        inline std::string hostname(){
#if defined(_WIN32)
            const char* name = std::getenv("COMPUTERNAME");
            return (name != nullptr && name[0] != '\0') ? name : "unknown-host";
#else
            char buffer[256];
            if(gethostname(buffer, sizeof(buffer)) != 0){
                return "unknown-host";
            }
            buffer[sizeof(buffer) - 1] = '\0';
            return buffer;
#endif
        }
        inline std::string default_run_id(){
            std::time_t now = std::time(nullptr);
            std::tm time_components{};
#if defined(_WIN32)
            localtime_s(&time_components, &now);
#else
            localtime_r(&now, &time_components);
#endif
            char timestamp[32];
            std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &time_components);
            return std::string(timestamp) + "_" + hostname() + "_" + std::to_string(process_id()) + "_" + random_hex(8);
        }
        inline std::string command_output(const char* command){
#if defined(_WIN32)
            FILE* pipe = _popen(command, "r");
#else
            FILE* pipe = popen(command, "r");
#endif
            if(pipe == nullptr){
                return "";
            }
            char buffer[128];
            std::string output;
            while(std::fgets(buffer, sizeof(buffer), pipe) != nullptr){
                output += buffer;
            }
#if defined(_WIN32)
            _pclose(pipe);
#else
            pclose(pipe);
#endif
            while(!output.empty() && (output.back() == '\n' || output.back() == '\r')){
                output.pop_back();
            }
            return output;
        }
        inline bool is_commit_hash(const std::string& value){
            if(value.size() != 40){
                return false;
            }
            for(char character: value){
                bool valid = (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
                if(!valid){
                    return false;
                }
            }
            return true;
        }
        inline bool is_unix_seconds(const std::string& value){
            if(value.empty()){
                return false;
            }
            for(char character: value){
                if(character < '0' || character > '9'){
                    return false;
                }
            }
            return true;
        }
        inline std::string detect_commit(){
            const char* environment_commit = std::getenv("METRA_COMMIT");
            if(environment_commit != nullptr && environment_commit[0] != '\0'){
                return environment_commit;
            }
#if defined(_WIN32)
            std::string output = command_output("git rev-parse HEAD 2>NUL");
#else
            std::string output = command_output("git rev-parse HEAD 2>/dev/null");
#endif
            if(is_commit_hash(output)){
                return output;
            }
#if defined(METRA_COMMIT)
            if(is_commit_hash(METRA_STRINGIFY(METRA_COMMIT))){
                return METRA_STRINGIFY(METRA_COMMIT);
            }
#endif
            return "no-hash";
        }
        inline std::string detect_commit_time(){
            const char* environment_commit_time = std::getenv("METRA_COMMIT_TIME");
            if(environment_commit_time != nullptr && is_unix_seconds(environment_commit_time)){
                return environment_commit_time;
            }
#if defined(_WIN32)
            std::string output = command_output("git show -s --format=%ct HEAD 2>NUL");
#else
            std::string output = command_output("git show -s --format=%ct HEAD 2>/dev/null");
#endif
            if(is_unix_seconds(output)){
                return output;
            }
#if defined(METRA_COMMIT_TIME)
            if(is_unix_seconds(METRA_STRINGIFY(METRA_COMMIT_TIME))){
                return METRA_STRINGIFY(METRA_COMMIT_TIME);
            }
#endif
            return "";
        }
        inline std::string build_payload(const Config& config, const std::string& name, const std::string& value_json){
            std::string payload = "{\"commit\":\"" + json_escape(config.commit) + "\",";
            if(is_unix_seconds(config.commit_time)){
                payload += "\"commit_time\":" + config.commit_time + ",";
            }
            payload += "\"run\":\"" + json_escape(config.run) + "\",\"name\":\"" + json_escape(name) + "\",\"value\":" + value_json + "}";
            return payload;
        }
        inline bool post_json(const Config& config, const std::string& payload, std::string& error_output){
            std::error_code error_code;
            std::filesystem::path temporary_directory = std::filesystem::temp_directory_path(error_code);
            if(error_code){
                error_output = "metra: cannot determine temporary directory: " + error_code.message();
                return false;
            }
            std::filesystem::path payload_path = temporary_directory / ("metra_" + std::to_string(process_id()) + "_" + random_hex(8) + ".json");
            {
                std::ofstream payload_file(payload_path, std::ios::binary);
                payload_file << payload;
                if(!payload_file.good()){
                    error_output = "metra: failed to write payload file " + payload_path.string();
                    return false;
                }
            }
            std::string url = config.url + "/api/log";
            std::string quoted_payload_path, quoted_url, quoted_header;
            if(!shell_quote(payload_path.string(), quoted_payload_path)){
                std::filesystem::remove(payload_path, error_code);
                error_output = "metra: unsupported character in path: " + payload_path.string();
                return false;
            }
            if(!shell_quote(url, quoted_url)){
                std::filesystem::remove(payload_path, error_code);
                error_output = "metra: unsupported character in URL: " + url;
                return false;
            }
            shell_quote("Content-Type: application/json", quoted_header);
#if defined(_WIN32)
            const char* null_device = "NUL";
#else
            const char* null_device = "/dev/null";
#endif
            std::string command = "curl -f -s -S -m 5 --connect-timeout 2 -X POST -H " + quoted_header + " --data-binary @" + quoted_payload_path + " -o " + null_device + " " + quoted_url;
            int status = std::system(command.c_str());
            std::filesystem::remove(payload_path, error_code);
            if(status != 0){
                error_output = "metra: logging to " + url + " failed (curl exit status " + std::to_string(status) + ")";
                if(status == 127 || status == 32512){
                    error_output += ". Is curl installed and on the PATH?";
                }
                else if(status == 22 || status == 5632){
                    error_output += ". The server rejected the request; check the payload and the server version.";
                }
                else{
                    error_output += ". Check METRA_URL and network connectivity.";
                }
                return false;
            }
            return true;
        }
    }
    inline Config config_from_environment(){
        Config config;
        const char* url = std::getenv("METRA_URL");
        if(url != nullptr && url[0] != '\0'){
            config.url = url;
            if(config.url.back() == '/'){
                config.url.pop_back();
            }
        }
        config.commit = detail::detect_commit();
        const char* environment_commit = std::getenv("METRA_COMMIT");
        const char* environment_commit_time = std::getenv("METRA_COMMIT_TIME");
        if(environment_commit == nullptr || environment_commit[0] == '\0' || environment_commit_time != nullptr){ // an overridden commit must not be paired with the auto-detected HEAD time
            config.commit_time = detail::detect_commit_time();
        }
        const char* run = std::getenv("METRA_RUN");
        config.run = (run != nullptr && run[0] != '\0') ? run : detail::default_run_id();
        return config;
    }
    inline const Config& global_config(){
        static const Config config = config_from_environment(); // the static per-process run id
        return config;
    }
    inline bool log_raw(const Config& config, const std::string& name, const std::string& value_json, std::string& error_output){
        if(config.url.empty()){
            return true;
        }
        if(name.empty()){
            error_output = "metra: metric name must not be empty";
            return false;
        }
        return detail::post_json(config, detail::build_payload(config, name, value_json), error_output);
    }
    inline bool log(const Config& config, const std::string& name, double value, std::string& error_output){
        return log_raw(config, name, detail::json_number(value), error_output);
    }
    inline bool log(const Config& config, const std::string& name, const std::vector<double>& values, std::string& error_output){
        return log_raw(config, name, detail::json_array(values), error_output);
    }
    namespace detail{
        inline bool log_global(const std::string& name, const std::string& value_json){
            const Config& config = global_config();
            if(config.url.empty()){
                static const bool notified = []{
                    std::cerr << "metra: METRA_URL not set, metric logging disabled" << std::endl;
                    return true;
                }();
                (void)notified;
                return true;
            }
            static int consecutive_failures = 0; // races on this counter are benign (worst case a few extra attempts); atomics are avoided repo-wide
            if(consecutive_failures >= 3){
                return false;
            }
            std::string error_output;
            if(!log_raw(config, name, value_json, error_output)){
                consecutive_failures++;
                std::cerr << error_output << std::endl;
                if(consecutive_failures >= 3){
                    std::cerr << "metra: disabling metric logging after " << consecutive_failures << " consecutive failures" << std::endl;
                }
                return false;
            }
            consecutive_failures = 0;
            return true;
        }
    }
    inline bool log_raw(const std::string& name, const std::string& value_json){
        return detail::log_global(name, value_json);
    }
    inline bool log(const std::string& name, double value){
        return detail::log_global(name, detail::json_number(value));
    }
    inline bool log(const std::string& name, const std::vector<double>& values){
        return detail::log_global(name, detail::json_array(values));
    }
}

#endif

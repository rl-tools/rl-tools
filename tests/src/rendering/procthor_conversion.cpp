#include <rl_tools/rendering/datasets/procthor/conversion.h>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

namespace {
    namespace fs = std::filesystem;
    namespace conversion = rl_tools::rendering::datasets::procthor::conversion;
    using Json = nlohmann::json;

    void write_u32(std::ostream& stream, uint32_t value){
        for(unsigned int byte = 0; byte < 4; byte++){
            stream.put(static_cast<char>(value >> (8 * byte)));
        }
    }

    uint32_t read_u32(std::istream& stream){
        uint32_t value = 0;
        for(unsigned int byte = 0; byte < 4; byte++){
            value |= static_cast<uint32_t>(static_cast<unsigned char>(stream.get())) << (8 * byte);
        }
        return value;
    }

    void write_glb(const fs::path& path, const Json& document){
        std::string data = document.dump();
        data.append((4 - data.size() % 4) % 4, ' ');
        std::ofstream stream(path, std::ios::binary);
        ASSERT_TRUE(stream.is_open());
        write_u32(stream, 0x46546c67);
        write_u32(stream, 2);
        write_u32(stream, static_cast<uint32_t>(20 + data.size()));
        write_u32(stream, static_cast<uint32_t>(data.size()));
        write_u32(stream, 0x4e4f534a);
        stream.write(data.data(), static_cast<std::streamsize>(data.size()));
        ASSERT_TRUE(stream.good());
    }

    class ProcTHORConversion: public ::testing::TestWithParam<std::tuple<bool, bool>>{
    protected:
        fs::path directory;

        void SetUp() override{
            const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
            directory = fs::temp_directory_path() / "rl_tools_procthor_conversion_tests" / info->test_suite_name() / info->name();
            fs::remove_all(directory);
            for(const auto* subdirectory : {"assets", "configs", "stages", "objects"}){
                fs::create_directories(directory / subdirectory);
            }
        }

        void TearDown() override{
            fs::remove_all(directory);
        }
    };

    TEST_P(ProcTHORConversion, serialized_light_indices){
        const auto [hssd, normalize] = GetParam();
        Json model = {
            {"asset", {{"version", "2.0"}}}, {"scene", 0},
            {"scenes", {{{"nodes", {0, 1, 2}}}}},
            {"nodes", Json::array()}, {"extensionsUsed", {"KHR_lights_punctual"}},
            {"extensions", {{"KHR_lights_punctual", {{"lights", {
                {{"type", "point"}, {"intensity", 2.0}, {"color", {1.0, 0.0, 0.0}}},
                {{"type", "spot"}, {"intensity", 7.0}, {"color", {0.0, 1.0, 0.0}},
                 {"spot", {{"innerConeAngle", 0.1}, {"outerConeAngle", 0.4}}}}
            }}}}}}
        };
        for(int light : {1, 0, 1}){
            model["nodes"].push_back({{"name", "lamp"}, {"translation", {0.0, 1.0, 2.0}},
                                      {"extensions", {{"KHR_lights_punctual", {{"light", light}}}}}});
        }
        write_glb(directory / "assets/stage.glb", model);
        write_glb(directory / "assets/object.glb", model);
        write_glb(directory / "stages/stage.glb", model);
        write_glb(directory / "objects/object.glb", model);
        const Json scene = {
            {"stage_instance", {{"template_name", "stage"}}},
            {"object_instances", {{{"template_name", hssd ? "objects/object" : "object"}, {"translation", {3.0, 4.0, 5.0}}}}},
            {"default_lighting", "no_lights"}
        };
        const auto input = directory / "configs/test.scene_instance.json";
        std::ofstream(input) << scene.dump();
        conversion::Options options;
        options.scene_instance_path = input.string();
        options.output_glb_path = (directory / "output.glb").string();
        options.normalize = normalize;
        options.dataset = hssd ? conversion::Dataset::HSSD : conversion::Dataset::AI2THOR_HAB;
        std::string error;
        ASSERT_TRUE(conversion::scene_instance_to_glb(options, error)) << error;

        std::ifstream stream(options.output_glb_path, std::ios::binary);
        ASSERT_EQ(read_u32(stream), 0x46546c67u);
        ASSERT_EQ(read_u32(stream), 2u);
        read_u32(stream);
        const auto length = read_u32(stream);
        ASSERT_EQ(read_u32(stream), 0x4e4f534au);
        std::string data(length, '\0');
        stream.read(data.data(), length);
        ASSERT_TRUE(stream.good());
        const auto output = Json::parse(data);
        const auto environment_path = Json::json_pointer("/scenes/0/extras/rl_tools/environment");
        if(hssd){
            EXPECT_FALSE(output.contains(environment_path));
        }
        else{
            ASSERT_TRUE(output.contains(environment_path));
            EXPECT_EQ(output.at(environment_path), (Json{{"mode", "solid"}, {"ambient", {0.6, 0.6, 0.6}}, {"background", {0.0, 0.0, 0.0}}}));
        }
        const auto& lights = output.at("extensions").at("KHR_lights_punctual").at("lights");
        ASSERT_EQ(lights.size(), 4u);
        std::vector<int> references;
        for(const auto& node : output.at("nodes")){
            if(node.contains("extensions") && node["extensions"].contains("KHR_lights_punctual")){
                const int index = node["extensions"]["KHR_lights_punctual"]["light"].get<int>();
                references.push_back(index);
                EXPECT_EQ(lights.at(index).at("intensity"), index % 2 == 0 ? 2.0 : 7.0);
            }
        }
        EXPECT_EQ(references, (std::vector<int>{1, 0, 1, 3, 2, 3}));
        EXPECT_EQ(lights[2]["color"], Json({1.0, 0.0, 0.0}));
        EXPECT_EQ(lights[3]["spot"], model["extensions"]["KHR_lights_punctual"]["lights"][1]["spot"]);
        EXPECT_EQ(output["nodes"].back()["matrix"][12], 3.0);
        EXPECT_EQ(output["nodes"].back()["matrix"][13], 4.0);
        EXPECT_EQ(output["nodes"].back()["matrix"][14], 5.0);
    }

    INSTANTIATE_TEST_SUITE_P(Datasets, ProcTHORConversion, ::testing::Combine(::testing::Bool(), ::testing::Bool()));
}

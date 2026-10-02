#include <rl_tools/rendering/datasets/procthor/conversion.h>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
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

    void write_glb(const fs::path& path, const Json& document, const std::vector<unsigned char>& binary = {}){
        std::string data = document.dump();
        data.append((4 - data.size() % 4) % 4, ' ');
        std::ofstream stream(path, std::ios::binary);
        ASSERT_TRUE(stream.is_open());
        write_u32(stream, 0x46546c67);
        write_u32(stream, 2);
        const size_t binary_size = (binary.size() + 3) / 4 * 4;
        write_u32(stream, static_cast<uint32_t>(20 + data.size() + (binary.empty() ? 0 : 8 + binary_size)));
        write_u32(stream, static_cast<uint32_t>(data.size()));
        write_u32(stream, 0x4e4f534a);
        stream.write(data.data(), static_cast<std::streamsize>(data.size()));
        if(!binary.empty()){
            write_u32(stream, static_cast<uint32_t>(binary_size));
            write_u32(stream, 0x004e4942);
            stream.write(reinterpret_cast<const char*>(binary.data()), static_cast<std::streamsize>(binary.size()));
            for(size_t byte = binary.size(); byte < binary_size; byte++){
                stream.put('\0');
            }
        }
        ASSERT_TRUE(stream.good());
    }

    struct Glb{
        Json document;
        std::vector<unsigned char> binary;
    };

    Glb read_glb(const fs::path& path){
        std::ifstream stream(path, std::ios::binary);
        EXPECT_EQ(read_u32(stream), 0x46546c67u);
        EXPECT_EQ(read_u32(stream), 2u);
        read_u32(stream);
        const auto length = read_u32(stream);
        EXPECT_EQ(read_u32(stream), 0x4e4f534au);
        std::string data(length, '\0');
        stream.read(data.data(), length);
        Glb result{Json::parse(data), {}};
        if(stream.peek() != std::char_traits<char>::eof()){
            result.binary.resize(read_u32(stream));
            EXPECT_EQ(read_u32(stream), 0x004e4942u);
            stream.read(reinterpret_cast<char*>(result.binary.data()), static_cast<std::streamsize>(result.binary.size()));
        }
        return result;
    }

    void append_floats(std::vector<unsigned char>& binary, const std::vector<float>& values){
        for(float value : values){
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            for(unsigned byte = 0; byte < 4; byte++){
                binary.push_back(static_cast<unsigned char>(bits >> (8 * byte)));
            }
        }
    }

    Glb triangle(){
        Glb model;
        append_floats(model.binary, {0, 0, 0, 1, 0, 0, 0, 1, 0});
        model.document = {
            {"asset", {{"version", "2.0"}}}, {"scene", 0}, {"scenes", {{{"nodes", {0}}}}},
            {"nodes", {{{"mesh", 0}}}},
            {"meshes", {{{"primitives", {{{"attributes", {{"POSITION", 0}}}}}}}}},
            {"buffers", {{{"byteLength", model.binary.size()}}}},
            {"bufferViews", {{{"buffer", 0}, {"byteLength", model.binary.size()}, {"target", 34962}}}},
            {"accessors", {{{"bufferView", 0}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"}, {"min", {0, 0, 0}}, {"max", {1, 1, 0}}}}}
        };
        return model;
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
            std::ofstream(directory / "configs/object_semantic_id_mapping.json") << Json{{"Undefined", 0}, {"Floor", 159}, {"Wall", 162}, {"Chair", 21}};
            std::ofstream(directory / "configs/object.object_config.json") << Json{{"semantic_id", 21}};
        }

        void TearDown() override{
            fs::remove_all(directory);
        }

        Glb convert(const Glb& model, bool add_object = false){
            const auto [hssd, normalize] = GetParam();
            write_glb(directory / (hssd ? "stages/stage.glb" : "assets/stage.glb"), model.document, model.binary);
            Json scene = {{"stage_instance", {{"template_name", "stage"}}}, {"default_lighting", "no_lights"}};
            if(add_object){
                write_glb(directory / (hssd ? "objects/object.glb" : "assets/object.glb"), model.document, model.binary);
                scene["object_instances"] = {{{"template_name", hssd ? "objects/object" : "object"}}};
            }
            const auto input = directory / "configs/test.scene_instance.json";
            std::ofstream(input) << scene.dump();
            conversion::Options options;
            options.scene_instance_path = input.string();
            options.output_glb_path = (directory / "output.glb").string();
            options.normalize = normalize;
            options.dataset = hssd ? conversion::Dataset::HSSD : conversion::Dataset::AI2THOR_HAB;
            std::string error;
            EXPECT_TRUE(conversion::scene_instance_to_glb(options, error)) << error;
            return read_glb(options.output_glb_path);
        }
    };

    TEST_P(ProcTHORConversion, exports_dataset_semantics_and_tracks_changes){
        auto model = triangle();
        model.document["nodes"][0]["extras"] = {{"type", "Floor"}};
        const auto result = convert(model, true);
        const auto& scene = result.document["scenes"][0];
        if(std::get<0>(GetParam())){
            EXPECT_FALSE(scene.contains(Json::json_pointer("/extras/rl_tools/segmentation")));
            return;
        }
        const auto& semantics = scene["extras"]["rl_tools"]["segmentation"];
        EXPECT_EQ(semantics["taxonomy"], "ai2thor-hab");
        EXPECT_EQ(semantics["classes"]["Floor"], 159);
        EXPECT_TRUE(semantics["classes"]["Floor"].is_number_integer());
        EXPECT_FALSE(semantics["classes"].contains("Ceiling"));
        ASSERT_EQ(scene["nodes"].size(), 2u);
        EXPECT_EQ(result.document["nodes"][scene["nodes"][0].get<size_t>()]["extras"]["rl_tools"]["segmentation_class"], 159);
        EXPECT_EQ(result.document["nodes"][scene["nodes"][1].get<size_t>()]["extras"]["rl_tools"]["segmentation_class"], 21);
        conversion::Options options;
        options.scene_instance_path = (directory / "configs/test.scene_instance.json").string();
        std::string first, second, third, error;
        ASSERT_TRUE(conversion::semantic_source(options, first, error)) << error;
        std::ofstream(directory / "configs/object.object_config.json") << Json{{"semantic_id", 162}};
        ASSERT_TRUE(conversion::semantic_source(options, second, error)) << error;
        EXPECT_NE(first, second);
        model.document["nodes"][0]["extras"]["type"] = "Wall";
        write_glb(directory / "assets/stage.glb", model.document, model.binary);
        ASSERT_TRUE(conversion::semantic_source(options, third, error)) << error;
        EXPECT_NE(second, third);
        second = third;
        std::ofstream(directory / "configs/object_semantic_id_mapping.json") << Json{{"Undefined", 0}, {"RenamedWall", 162}};
        ASSERT_TRUE(conversion::semantic_source(options, third, error)) << error;
        EXPECT_NE(second, third);
        std::ofstream(directory / "configs/object.object_config.json") << Json{{"semantic_id", 999}};
        EXPECT_FALSE(conversion::semantic_source(options, third, error));
    }

    TEST_P(ProcTHORConversion, unknown_structural_category_is_zero){
        auto model = triangle();
        model.document["nodes"][0]["extras"] = {{"type", "Ceiling"}};
        const auto result = convert(model);
        if(!std::get<0>(GetParam())){
            const auto root = result.document["scenes"][0]["nodes"][0].get<size_t>();
            EXPECT_EQ(result.document["nodes"][root]["extras"]["rl_tools"]["segmentation_class"], 0);
        }
    }

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

    TEST_P(ProcTHORConversion, split_roots_keep_order_and_transforms_without_parent_links){
        auto model = triangle();
        model.document["nodes"] = {
            {{"name", "level"}, {"translation", {1, 2, 3}}, {"children", {1}}},
            {{"name", "room"}, {"translation", {4, 5, 6}}, {"children", {2, 3}}},
            {{"name", "wall"}, {"translation", {7, 8, 9}}, {"children", {4}}},
            {{"name", "floor"}, {"translation", {10, 11, 12}}, {"children", {5}}},
            {{"mesh", 0}, {"translation", {0, 0, 2}}},
            {{"mesh", 0}, {"translation", {0, 0, 4}}}
        };
        const auto output = convert(model).document;
        ASSERT_EQ(output["scenes"][0]["nodes"], Json({2, 3}));
        EXPECT_EQ(output["nodes"][2]["name"], "wall");
        EXPECT_EQ(output["nodes"][3]["name"], "floor");
        for(const auto& node : output["nodes"]){
            for(int child : node.value("children", std::vector<int>{})){
                EXPECT_NE(child, 2);
                EXPECT_NE(child, 3);
            }
        }
        for(int component = 0; component < 3; component++){
            EXPECT_EQ(output["nodes"][2]["matrix"][12 + component], 12 + component * 3);
            EXPECT_EQ(output["nodes"][3]["matrix"][12 + component], 15 + component * 3);
        }
        EXPECT_EQ(output["nodes"][2]["children"], Json({4}));
        EXPECT_EQ(output["nodes"][3]["children"], Json({5}));
        EXPECT_EQ(output["nodes"][4]["translation"], Json({0, 0, 2}));
        EXPECT_EQ(output["nodes"][5]["translation"], Json({0, 0, 4}));
    }

    TEST_P(ProcTHORConversion, merged_buffers_align_accessors_and_preserve_bytes){
        auto model = triangle();
        model.binary.push_back(123);
        model.document["buffers"][0]["byteLength"] = model.binary.size();
        const auto output = convert(model, true);
        ASSERT_EQ(output.document["accessors"].size(), 2u);
        ASSERT_EQ(output.document["buffers"].size(), 1u);
        for(size_t index = 0; index < 2; index++){
            const auto& accessor = output.document["accessors"][index];
            const auto& view = output.document["bufferViews"][accessor["bufferView"].get<size_t>()];
            const size_t offset = view.value("byteOffset", size_t{0}) + accessor.value("byteOffset", size_t{0});
            EXPECT_EQ(offset % sizeof(float), 0u);
            ASSERT_LE(offset + model.binary.size(), output.binary.size());
            EXPECT_EQ(std::vector<unsigned char>(output.binary.begin() + offset, output.binary.begin() + offset + model.binary.size()), model.binary);
        }
    }

    TEST_P(ProcTHORConversion, material_repairs_preserve_valid_textures_and_emission){
        auto model = triangle();
        const size_t uv_offset = model.binary.size();
        append_floats(model.binary, {0, 0, 1, 0, 0, 1});
        model.document["bufferViews"].push_back({{"buffer", 0}, {"byteOffset", uv_offset}, {"byteLength", 24}, {"target", 34962}});
        model.document["accessors"].push_back({{"bufferView", 1}, {"componentType", 5126}, {"count", 3}, {"type", "VEC2"}});
        const unsigned char png[] = {
            137,80,78,71,13,10,26,10,0,0,0,13,73,72,68,82,0,0,0,1,0,0,0,1,8,6,0,0,0,31,21,196,137,
            0,0,0,11,73,68,65,84,120,156,99,248,15,4,0,9,251,3,253,251,94,107,43,0,0,0,0,73,69,78,68,174,66,96,130
        };
        const size_t image_offset = model.binary.size();
        model.binary.insert(model.binary.end(), std::begin(png), std::end(png));
        model.document["bufferViews"].push_back({{"buffer", 0}, {"byteOffset", image_offset}, {"byteLength", sizeof(png)}});
        model.document["buffers"][0]["byteLength"] = model.binary.size();
        model.document["images"] = {{{"bufferView", 2}, {"mimeType", "image/png"}}};
        model.document["textures"] = {{{"source", 0}}};
        model.document["extensionsUsed"] = {"KHR_materials_emissive_strength", "KHR_texture_transform"};
        model.document["materials"] = {{{"pbrMetallicRoughness", {
            {"baseColorFactor", {4, -0.25, 0.5, 0.8}},
            {"baseColorTexture", {{"index", 0}}}, {"metallicRoughnessTexture", {{"index", 0}}}
        }}, {"normalTexture", {{"index", 0}, {"scale", 2.0}}},
            {"occlusionTexture", {{"index", 0}, {"extensions", {{"KHR_texture_transform", {{"texCoord", 1}}}}}}},
            {"emissiveTexture", {{"index", 0}, {"texCoord", 1}}}, {"emissiveFactor", {0.8, 0.4, 0.2}},
            {"extensions", {{"KHR_materials_emissive_strength", {{"emissiveStrength", 4}}}}}
        }};
        model.document["materials"].push_back(model.document["materials"][0]);
        model.document["nodes"] = {{{"mesh", 0}}, {{"mesh", 1}}, {{"mesh", 2}}, {{"mesh", 3}}};
        model.document["scenes"][0]["nodes"] = {0, 1, 2, 3};
        model.document["meshes"] = Json::array();
        for(int mesh = 0; mesh < 4; mesh++){
            Json attributes = {{"POSITION", 0}};
            if(mesh != 0){
                attributes["TEXCOORD_0"] = 1;
            }
            if(mesh == 2){
                attributes["TEXCOORD_1"] = 1;
            }
            model.document["meshes"].push_back({{"primitives", {{{"attributes", attributes}, {"material", 0}}}}});
        }
        const auto output = convert(model).document;
        ASSERT_EQ(output["materials"].size(), 4u);
        for(const auto& material : output["materials"]){
            EXPECT_EQ(material["pbrMetallicRoughness"]["baseColorFactor"], Json({1, 0, 0.5, 0.8}));
            EXPECT_EQ(material["emissiveFactor"], model.document["materials"][0]["emissiveFactor"]);
            EXPECT_EQ(material["extensions"], model.document["materials"][0]["extensions"]);
            if(std::get<1>(GetParam()) && material.contains("occlusionTexture")){
                EXPECT_FALSE(material["occlusionTexture"].contains("extensions"));
            }
        }
        const auto material_for = [&](size_t mesh) -> const Json& {
            return output["materials"][output["meshes"][mesh]["primitives"][0]["material"].get<size_t>()];
        };
        for(int mesh = 0; mesh < 4; mesh++){
            const auto& material = material_for(mesh);
            EXPECT_EQ(material["pbrMetallicRoughness"].contains("baseColorTexture"), mesh != 0);
            EXPECT_EQ(material["pbrMetallicRoughness"].contains("metallicRoughnessTexture"), mesh != 0);
            EXPECT_EQ(material.contains("normalTexture"), mesh != 0);
            EXPECT_EQ(material.contains("occlusionTexture"), mesh == 2);
            EXPECT_EQ(material.contains("emissiveTexture"), mesh == 2);
        }
        EXPECT_EQ(output["meshes"][1]["primitives"][0]["material"], output["meshes"][3]["primitives"][0]["material"]);
    }

    INSTANTIATE_TEST_SUITE_P(Datasets, ProcTHORConversion, ::testing::Combine(::testing::Bool(), ::testing::Bool()));
}

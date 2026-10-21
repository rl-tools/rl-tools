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

    const char* const HSSD_CONDENSED = R"(Object Hash,"Articulated Object:
does the object have links/joint/slides?","Is Pickable:
is the object small enough to be picked up?","Semantic Category:
CONDENSED

condensed by a couple hundred, ""quoted""","Semantic Category:
The primary semantic category for the object",,
object,No,Yes,chair,armchair,,
condensed,No,Yes,chair,armchair,,
base,No,No,table,"dining_table, oak",,
unlexiconed,No,#N/A,shelf/cabinet,shelf,,
)";

    const char* const HSSD_OBJECTS = R"(id,name,wnsynsetkey,main_category,super_category
condensed,"Chair, ""oak""",chair.n.01,table,furniture
fallback,Door,door.n.01,door,opening
unlexiconed,Shelf,shelf.n.01,,
)";

    const Json HSSD_REGIONS = {{{"name", "kitchen"}, {"label", "kitchen"}, {"poly_loop", {{0.5, 0.0, -1.25}, {2.0, 0.0, -1.25}, {2.0, 0.0, 1.0}}}, {"floor_height", 0.0}, {"extrusion_height", 2.8}}};

    class ProcTHORConversion: public ::testing::TestWithParam<std::tuple<bool, bool>>{
    protected:
        fs::path directory;

        void SetUp() override{
            const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
            directory = fs::temp_directory_path() / "rl_tools_procthor_conversion_tests" / info->test_suite_name() / info->name();
            fs::remove_all(directory);
            for(const auto* subdirectory : {"assets", "configs", "stages", "objects", "metadata", "semantics/scenes"}){
                fs::create_directories(directory / subdirectory);
            }
            std::ofstream(directory / "configs/object_semantic_id_mapping.json") << Json{{"Undefined", 0}, {"Floor", 159}, {"Wall", 162}, {"Chair", 21}};
            std::ofstream(directory / "configs/object.object_config.json") << Json{{"semantic_id", 21}};
            std::ofstream(directory / "semantics/hssd-hab_semantic_lexicon.json") << Json{{"classes", {{{"id", 0}, {"name", "unknown"}}, {{"id", 1}, {"name", "chair"}}, {{"id", 2}, {"name", "door"}}, {{"id", 3}, {"name", "floor"}}, {{"id", 4}, {"name", "table"}}}}};
            std::ofstream(directory / "metadata/hssd_obj_semantics_condensed.csv") << HSSD_CONDENSED;
            std::ofstream(directory / "semantics/objects.csv") << HSSD_OBJECTS;
            std::ofstream(directory / "semantics/scenes/test.semantic_config.json") << Json{{"region_annotations", HSSD_REGIONS}};
        }

        void TearDown() override{
            fs::remove_all(directory);
        }

        bool hssd() const{
            return std::get<0>(GetParam());
        }

        // HSSD configs carry a stale semantic_id (3 = floor in the fixture lexicon) that labels must ignore
        void write_object(const std::string& name, const Glb& model){
            write_glb(directory / (hssd() ? "objects" : "assets") / (name + ".glb"), model.document, model.binary);
            if(hssd()){
                std::ofstream(directory / "objects" / (name + ".object_config.json")) << Json{{"semantic_id", 3}};
            }
        }

        bool convert_scene(const std::vector<std::string>& objects, std::string& error){
            Json scene = {{"stage_instance", {{"template_name", "stage"}}}, {"default_lighting", "no_lights"}, {"object_instances", Json::array()}};
            for(const auto& object : objects){
                scene["object_instances"].push_back({{"template_name", hssd() ? "objects/" + object : object}});
            }
            const auto input = directory / "configs/test.scene_instance.json";
            std::ofstream(input) << scene.dump();
            conversion::Options options;
            options.scene_instance_path = input.string();
            options.output_glb_path = (directory / "output.glb").string();
            options.normalize = std::get<1>(GetParam());
            options.dataset = hssd() ? conversion::Dataset::HSSD : conversion::Dataset::AI2THOR_HAB;
            return conversion::scene_instance_to_glb(options, error);
        }

        Glb convert(const Glb& model, size_t instances = 0){
            write_glb(directory / (hssd() ? "stages/stage.glb" : "assets/stage.glb"), model.document, model.binary);
            if(instances > 0){
                write_object("object", model);
            }
            std::string error;
            EXPECT_TRUE(convert_scene(std::vector<std::string>(instances, "object"), error)) << error;
            return read_glb(directory / "output.glb");
        }
    };

    TEST_P(ProcTHORConversion, exports_dataset_semantics_and_tracks_changes){
        auto model = triangle();
        model.document["nodes"][0]["extras"] = {{"type", "Floor"}};
        const auto result = convert(model, 1);
        const auto& scene = result.document["scenes"][0];
        if(hssd()){
            EXPECT_EQ(scene["extras"]["rl_tools"]["segmentation"]["taxonomy"], "hssd-hab");
            ASSERT_EQ(scene["nodes"].size(), 2u);
            EXPECT_EQ(result.document["nodes"][scene["nodes"][0].get<size_t>()]["extras"]["rl_tools"]["segmentation_class"], 0);
            EXPECT_EQ(result.document["nodes"][scene["nodes"][1].get<size_t>()]["extras"]["rl_tools"]["segmentation_class"], 1);
            conversion::Options options;
            options.scene_instance_path = (directory / "configs/test.scene_instance.json").string();
            options.dataset = conversion::Dataset::HSSD;
            std::string first, second, error;
            ASSERT_TRUE(conversion::semantic_source(options, first, error)) << error;
            std::ofstream(directory / "metadata/hssd_obj_semantics_condensed.csv") << std::string(HSSD_CONDENSED).replace(std::string(HSSD_CONDENSED).find("object,No,Yes,chair"), 19, "object,No,Yes,table");
            ASSERT_TRUE(conversion::semantic_source(options, second, error)) << error;
            EXPECT_NE(first, second);
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
        const auto root = result.document["scenes"][0]["nodes"][0].get<size_t>();
        EXPECT_EQ(result.document["nodes"][root]["extras"]["rl_tools"]["segmentation_class"], 0);
    }

    TEST_P(ProcTHORConversion, hssd_labels_follow_dataset_tables_and_keep_records){
        if(!hssd()){
            return;
        }
        auto stage = triangle();
        stage.document["nodes"][0]["extras"] = {{"type", "floor"}};
        write_glb(directory / "stages/stage.glb", stage.document, stage.binary);
        for(const auto* name : {"condensed", "fallback", "unlexiconed", "base_part_1", "none"}){
            write_object(name, triangle());
        }
        std::string error;
        ASSERT_TRUE(convert_scene({"condensed", "missing", "fallback", "unlexiconed", "base_part_1", "none", "condensed"}, error)) << error;
        const auto output = read_glb(directory / "output.glb").document;
        const auto& scene = output["scenes"][0];
        EXPECT_EQ(scene["extras"]["rl_tools"]["segmentation"], (Json{{"taxonomy", "hssd-hab"}, {"classes", {{"unknown", 0}, {"chair", 1}, {"door", 2}, {"floor", 3}, {"table", 4}}}}));
        EXPECT_EQ(scene["extras"]["hssd"], (Json{{"scene", "test"}, {"region_annotations", HSSD_REGIONS}}));
        const auto& environment = scene["extras"]["rl_tools"]["environment"];
        EXPECT_EQ(environment["mode"], "solid");
        EXPECT_EQ(environment["background"], Json({0.0, 0.0, 0.0}));
        EXPECT_GT(environment["ambient"][0].get<double>(), 0.0);
        EXPECT_EQ(environment["ambient"], Json({environment["ambient"][0], environment["ambient"][0], environment["ambient"][0]}));
        const std::vector<std::tuple<std::string, int, int, Json>> expected = {
            {"condensed", 0, 1, {{"wnsynsetkey", "chair.n.01"}, {"main_category", "table"}, {"super_category", "furniture"}, {"condensed_category", "chair"}, {"primary_category", "armchair"}, {"pickable", true}}},
            {"fallback", 2, 2, {{"wnsynsetkey", "door.n.01"}, {"main_category", "door"}, {"super_category", "opening"}}},
            {"unlexiconed", 3, 0, {{"wnsynsetkey", "shelf.n.01"}, {"condensed_category", "shelf/cabinet"}, {"primary_category", "shelf"}}},
            {"base_part_1", 4, 4, {{"condensed_category", "table"}, {"primary_category", "dining_table, oak"}, {"pickable", false}}},
            {"none", 5, 0, Json::object()},
            {"condensed", 6, 1, {{"wnsynsetkey", "chair.n.01"}, {"main_category", "table"}, {"super_category", "furniture"}, {"condensed_category", "chair"}, {"primary_category", "armchair"}, {"pickable", true}}}
        };
        ASSERT_EQ(scene["nodes"].size(), 1 + expected.size());
        EXPECT_EQ(output["nodes"][scene["nodes"][0].get<size_t>()]["extras"]["rl_tools"]["segmentation_class"], 3);
        for(size_t object = 0; object < expected.size(); object++){
            const auto& [name, instance, id, categories] = expected[object];
            const auto& node = output["nodes"][scene["nodes"][object + 1].get<size_t>()];
            EXPECT_EQ(node["name"], name);
            EXPECT_EQ(node["extras"]["rl_tools"]["segmentation_class"], id) << name;
            Json record = {{"template", "objects/" + name}, {"instance", instance}};
            if(!categories.empty()){
                record["categories"] = categories;
            }
            EXPECT_EQ(node["extras"]["hssd"], record) << name;
        }
    }

    TEST_P(ProcTHORConversion, hssd_requires_semantic_config_and_object_configs){
        if(!hssd()){
            return;
        }
        convert(triangle(), 1);
        std::string error;
        fs::remove(directory / "objects/object.object_config.json");
        EXPECT_FALSE(convert_scene({"object"}, error));
        EXPECT_NE(error.find("objects/object"), std::string::npos) << error;
        write_object("object", triangle());
        fs::remove(directory / "semantics/scenes/test.semantic_config.json");
        EXPECT_FALSE(convert_scene({"object"}, error));
        EXPECT_NE(error.find("test.semantic_config.json"), std::string::npos) << error;
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
        std::ofstream(directory / "objects/object.object_config.json") << Json::object();
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
        ASSERT_TRUE(output.contains(environment_path));
        if(!hssd){
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
        model.document["bufferViews"].push_back({{"buffer", 0}, {"byteOffset", 36}, {"byteLength", 1}});
        auto object = triangle();
        object.binary.clear();
        append_floats(object.binary, {0, 0, 0, 2, 0, 0, 0, 2, 0});
        write_object("object", object);
        write_glb(directory / (hssd() ? "stages/stage.glb" : "assets/stage.glb"), model.document, model.binary);
        std::string error;
        ASSERT_TRUE(convert_scene({"object"}, error)) << error;
        const auto output = read_glb(directory / "output.glb");
        ASSERT_EQ(output.document["accessors"].size(), 2u);
        ASSERT_EQ(output.document["buffers"].size(), 1u);
        for(size_t index = 0; index < 2; index++){
            const auto& expected = index == 0 ? model.binary : object.binary;
            const auto& accessor = output.document["accessors"][index];
            const auto& view = output.document["bufferViews"][accessor["bufferView"].get<size_t>()];
            const size_t offset = view.value("byteOffset", size_t{0}) + accessor.value("byteOffset", size_t{0});
            EXPECT_EQ(offset % sizeof(float), 0u);
            ASSERT_LE(offset + 36, output.binary.size());
            EXPECT_EQ(std::vector<unsigned char>(output.binary.begin() + offset, output.binary.begin() + offset + 36), std::vector<unsigned char>(expected.begin(), expected.begin() + 36));
        }
        const auto& odd = output.document["bufferViews"][1];
        EXPECT_EQ(output.binary.at(odd.value("byteOffset", size_t{0})), 123);
    }

    TEST_P(ProcTHORConversion, identical_buffer_views_are_stored_once){
        const auto model = triangle();
        const auto output = convert(model, 2);
        ASSERT_EQ(output.document["accessors"].size(), 3u);
        EXPECT_EQ(output.binary.size(), model.binary.size());
        for(const auto& accessor : output.document["accessors"]){
            const auto& view = output.document["bufferViews"][accessor["bufferView"].get<size_t>()];
            const size_t offset = view.value("byteOffset", size_t{0}) + accessor.value("byteOffset", size_t{0});
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

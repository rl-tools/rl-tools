#include <rl_tools/operations/cpu.h>
#include <rl_tools/rendering/datasets/glb/operations_cpu.h>
#include <gtest/gtest.h>
#include <rl_tools/rendering/segmentation_cpu.h>
#include <rl_tools/rendering/datasets/procthor/operations_cpu.h>

#include <filesystem>
#include <fstream>
#include <optional>

namespace {
    namespace rlt = rl_tools;
    using Json = nlohmann::json;
    using Environment = rlt::rendering::Scene::Environment;
    using Roots = rlt::Mode<rlt::mode::rendering::SceneRoots<>>;
    using rlt::add;
    using rlt::load;
    using rlt::segmentation_taxonomy;
    using rlt::validate_semantic_segmentation;

    class GlbEnvironment: public ::testing::Test{
    protected:
        rlt::devices::DefaultCPU device;
        std::filesystem::path directory;
        const Json authored = {{"mode", "solid"}, {"ambient", {0.6, 0.3, 2.0}}, {"background", {0.02, 0.04, 0.08}}};

        void SetUp() override{
            rlt::init(device);
            directory = std::filesystem::temp_directory_path() / "rl_tools_glb_environment" / ::testing::UnitTest::GetInstance()->current_test_info()->name();
            std::filesystem::create_directories(directory);
        }

        void TearDown() override{
            std::filesystem::remove_all(directory);
        }

        std::string write_glb(std::optional<Json> environment, const Json& classes = Json(), bool single_root = false){
            const float positions[9] = {2, -1, -1, 2, 1, -1, 2, 0, 1};
            Json document = {
                {"asset", {{"version", "2.0"}}}, {"scene", 1},
                {"scenes", {{{"nodes", {0, 1}}, {"extras", {{"rl_tools", {{"environment", {{"mode", "invalid"}}}}}}}}, {{"nodes", {0, 1}}}}},
                {"nodes", {{{"name", "first"}, {"mesh", 0}}, {{"name", "second"}, {"mesh", 0}, {"translation", {0, 0, 4}}}}},
                {"meshes", {{{"primitives", {{{"attributes", {{"POSITION", 0}}}}}}}}},
                {"buffers", {{{"byteLength", sizeof(positions)}}}},
                {"bufferViews", {{{"buffer", 0}, {"byteLength", sizeof(positions)}}}},
                {"accessors", {{{"bufferView", 0}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"}, {"min", {2, -1, -1}}, {"max", {2, 1, 1}}}}}
            };
            if(environment.has_value()){
                document["scenes"][1]["extras"]["rl_tools"]["environment"] = std::move(*environment);
            }
            if(!classes.is_null()){
                document["scenes"][1]["extras"]["rl_tools"]["segmentation"] = {{"taxonomy", "test"}, {"classes", {{"Unknown", 0}, {"Chair", 21}, {"Wall", 162}}}};
                for(size_t i = 0; i < classes.size(); i++){
                    document["nodes"][i]["extras"]["rl_tools"]["segmentation_class"] = classes[i];
                }
            }
            if(single_root){
                document["scenes"][1]["nodes"] = {0};
            }
            auto encoded = document.dump();
            encoded.append((4 - encoded.size() % 4) % 4, ' ');
            const auto path = directory / "ProcTHOR-Train-1.glb";
            std::ofstream stream(path, std::ios::binary);
            const auto u32 = [&](uint32_t value){
                for(unsigned byte = 0; byte < 4; byte++){
                    stream.put(static_cast<char>(value >> (8 * byte)));
                }
            };
            u32(0x46546c67); u32(2); u32(static_cast<uint32_t>(28 + encoded.size() + sizeof(positions)));
            u32(static_cast<uint32_t>(encoded.size())); u32(0x4e4f534a);
            stream.write(encoded.data(), encoded.size());
            u32(sizeof(positions)); u32(0x004e4942);
            for(float position : positions){
                uint32_t bits;
                std::memcpy(&bits, &position, sizeof(bits));
                u32(bits);
            }
            return path.string();
        }

        void expect_authored(const Environment& environment){
            EXPECT_EQ(environment.mode, Environment::Mode::SOLID);
            for(int component = 0; component < 3; component++){
                EXPECT_FLOAT_EQ(environment.ambient[component], authored["ambient"][component].get<float>());
                EXPECT_FLOAT_EQ(environment.horizon[component], authored["background"][component].get<float>());
            }
        }
    };

    TEST_F(GlbEnvironment, semantic_roots_and_welded_rejection){
        const auto path = write_glb(authored, Json{21, 162});
        rlt::rendering::Bundle<float> roots, welded;
        ASSERT_TRUE(load(device, roots, path, Roots{}));
        ASSERT_TRUE(load(device, welded, path));
        ASSERT_EQ(roots.scene.objects.size(), 2u);
        EXPECT_EQ(roots.scene.objects[0].segmentation_class, 21u);
        EXPECT_EQ(roots.scene.objects[1].segmentation_class, 162u);
        auto taxonomy = segmentation_taxonomy(device, roots.scene);
        ASSERT_TRUE(taxonomy);
        EXPECT_EQ(taxonomy->name, "test");
        EXPECT_EQ(taxonomy->classes.at(21), "Chair");
        EXPECT_TRUE(roots.scene.objects[0].segmentation_class_valid);
        EXPECT_FALSE(welded.scene.objects[0].segmentation_class_valid);
        std::vector<const rlt::rendering::Object*> objects = {&welded.scene.objects[0]};
        EXPECT_THROW(validate_semantic_segmentation(device, objects), std::invalid_argument);
        rlt::rendering::ObjectAssembly assembly;
        ASSERT_TRUE(load(device, assembly, path));
        rlt::rendering::AssetPool pool;
        pool.assemblies.push_back(assembly);
        EXPECT_EQ(segmentation_taxonomy(device, pool.assemblies[0])->classes, taxonomy->classes);
        EXPECT_EQ(pool.assemblies[0].objects[1].segmentation_class, 162u);
    }

    TEST_F(GlbEnvironment, one_root_repeated_classes_and_unknown){
        for(const auto& classes : {Json{21, 21}, Json{0, 0}}){
            const auto path = write_glb(std::nullopt, classes);
            rlt::rendering::Object object;
            ASSERT_TRUE(load(device, object, path));
            EXPECT_TRUE(object.segmentation_class_valid);
            EXPECT_EQ(object.segmentation_class, classes[0]);
        }
        const auto path = write_glb(std::nullopt, Json{21}, true);
        rlt::rendering::Bundle<float> roots;
        ASSERT_TRUE(load(device, roots, path, Roots{}));
        ASSERT_EQ(roots.scene.objects.size(), 1u);
        EXPECT_EQ(roots.scene.objects[0].name, "first");
        EXPECT_EQ(roots.scene.objects[0].segmentation_class, 21u);
        EXPECT_FALSE(roots.scene.objects[0].meshes.empty());
    }

    TEST_F(GlbEnvironment, rejects_invalid_labels_and_taxonomy_conflicts){
        for(const auto& invalid : {Json(-1), Json(999), Json(0xFFFFFFFFu), Json("21")}){
            const auto path = write_glb(std::nullopt, Json{invalid, 21});
            rlt::rendering::Bundle<float> roots;
            EXPECT_FALSE(load(device, roots, path, Roots{}));
        }
        rlt::rendering::Object first, second;
        auto a = std::make_shared<rlt::rendering::SegmentationTaxonomy>();
        a->name = "a"; a->classes = {{0, "Unknown"}};
        auto b = std::make_shared<rlt::rendering::SegmentationTaxonomy>(*a);
        b->name = "b";
        first.segmentation_taxonomy = a; second.segmentation_taxonomy = b;
        std::vector<const rlt::rendering::Object*> objects = {&first, &second};
        EXPECT_THROW(validate_semantic_segmentation(device, objects), std::invalid_argument);
    }

    TEST_F(GlbEnvironment, glb_dataset_forwards_roots_mode){
        const auto path = write_glb(authored, Json{21, 162});
        rlt::rendering::datasets::procthor::GLB dataset;
        rlt::rendering::datasets::procthor::Corpus corpus;
        corpus.references.push_back(path);
        rlt::rendering::Bundle<float> roots;
        ASSERT_TRUE(load(device, dataset, corpus, 0, roots, Roots{}));
        ASSERT_EQ(roots.scene.instances.size(), 2u);
        EXPECT_EQ(roots.scene.objects[1].segmentation_class, 162u);
    }

#ifdef RL_TOOLS_TEST_PROCTHOR_CONVERSION
    TEST_F(GlbEnvironment, raw_dataset_forwards_mode_and_invalidates_semantic_cache){
        const auto source = write_glb(std::nullopt);
        std::filesystem::create_directories(directory / "assets");
        std::filesystem::create_directories(directory / "configs");
        std::filesystem::copy_file(source, directory / "assets/stage.glb");
        std::filesystem::copy_file(source, directory / "assets/object.glb");
        std::ofstream(directory / "configs/object_semantic_id_mapping.json") << Json{{"Undefined", 0}, {"Chair", 21}, {"Wall", 162}};
        std::ofstream(directory / "configs/object.object_config.json") << Json{{"semantic_id", 21}};
        const auto scene = directory / "configs/test.scene_instance.json";
        std::ofstream(scene) << Json{{"stage_instance", {{"template_name", "stage"}}}, {"object_instances", {{{"template_name", "object"}}}}};
        rlt::rendering::datasets::procthor::AI2ThorHab dataset;
        dataset.root = directory.string();
        dataset.cache_directory = (directory / "cache").string();
        rlt::rendering::datasets::procthor::Corpus corpus;
        corpus.references.push_back(scene.string());
        rlt::rendering::Bundle<float> first, second, third;
        ASSERT_TRUE(load(device, dataset, corpus, 0, first, Roots{}));
        ASSERT_EQ(first.scene.instances.size(), 3u);
        EXPECT_EQ(first.scene.objects.back().segmentation_class, 21u);
        std::ofstream(directory / "configs/object.object_config.json") << Json{{"semantic_id", 162}};
        ASSERT_TRUE(load(device, dataset, corpus, 0, second, Roots{}));
        EXPECT_EQ(second.scene.objects.back().segmentation_class, 162u);
        EXPECT_NE(first.metadata.content_hash, second.metadata.content_hash);
        ASSERT_TRUE(load(device, dataset, corpus, 0, third, Roots{}));
        EXPECT_EQ(second.metadata.content_hash, third.metadata.content_hash);
        size_t cached = 0;
        for(const auto& entry : std::filesystem::directory_iterator(dataset.cache_directory)){
            cached += entry.path().extension() == ".glb";
        }
        EXPECT_EQ(cached, 2u);
    }
#endif

    TEST_F(GlbEnvironment, full_scene_layouts_preserve_environment){
        const auto path = write_glb(authored);
        rlt::rendering::Bundle<float> welded, roots;
        ASSERT_TRUE(load(device, welded, path));
        ASSERT_TRUE(load(device, roots, path, Roots{}));
        expect_authored(welded.scene.environment);
        expect_authored(roots.scene.environment);
        EXPECT_EQ(welded.scene.instances.size(), 1u);
        ASSERT_EQ(roots.scene.instances.size(), 2u);
        EXPECT_EQ(roots.scene.objects[0].name, "first");
        EXPECT_EQ(roots.scene.objects[1].name, "second");
        EXPECT_EQ(roots.scene.instances[0].object, 0u);
        EXPECT_EQ(roots.scene.instances[1].object, 1u);
        EXPECT_EQ(welded.metadata.content_hash, roots.metadata.content_hash);
        for(int component = 0; component < 3; component++){
            EXPECT_FLOAT_EQ(welded.metadata.center[component], roots.metadata.center[component]);
            EXPECT_FLOAT_EQ(welded.metadata.half_extent[component], roots.metadata.half_extent[component]);
        }
        rlt::rendering::ObjectAssembly assembly;
        ASSERT_TRUE(load(device, assembly, path));
        for(size_t part = 0; part < assembly.parts.size(); part++){
            for(int element = 0; element < 12; element++){
                EXPECT_FLOAT_EQ(roots.scene.instances[part].transform[element], assembly.parts[part].transform[element]);
            }
        }
    }

    TEST_F(GlbEnvironment, absent_metadata_keeps_generic_defaults){
        const auto path = write_glb(std::nullopt);
        rlt::rendering::Bundle<float> welded, roots;
        ASSERT_TRUE(load(device, welded, path));
        ASSERT_TRUE(load(device, roots, path, Roots{}));
        for(const auto* bundle : {&welded, &roots}){
            EXPECT_EQ(bundle->scene.environment.mode, Environment::Mode::DEFAULT);
            for(float ambient : bundle->scene.environment.ambient){
                EXPECT_FLOAT_EQ(ambient, 0.1f);
            }
        }
    }

    TEST_F(GlbEnvironment, composition_preserves_parent_environment){
        const auto path = write_glb(authored);
        rlt::rendering::Bundle<float> bundle;
        bundle.scene.environment.mode = Environment::Mode::SOLID;
        bundle.scene.environment.ambient[0] = 0.25f;
        bundle.scene.environment.horizon[2] = 0.75f;
        ASSERT_TRUE(add(device, bundle, path));
        EXPECT_FLOAT_EQ(bundle.scene.environment.ambient[0], 0.25f);
        EXPECT_FLOAT_EQ(bundle.scene.environment.horizon[2], 0.75f);
    }

    TEST_F(GlbEnvironment, malformed_environment_rejects_scene_without_affecting_asset_import){
        std::vector<Json> invalid{nullptr, 1, Json::array(), authored, authored, authored, authored, authored};
        invalid[3]["mode"] = "gradient";
        invalid[4]["ambient"] = {0, 0};
        invalid[5]["ambient"][1] = -0.1;
        invalid[6]["background"][1] = 1e100;
        invalid[7].erase("background");
        for(const auto& environment : invalid){
            SCOPED_TRACE(environment.dump());
            const auto path = write_glb(environment);
            rlt::rendering::Bundle<float> welded, roots;
            EXPECT_FALSE(load(device, welded, path));
            EXPECT_FALSE(load(device, roots, path, Roots{}));
            EXPECT_TRUE(welded.scene.objects.empty());
            EXPECT_TRUE(roots.scene.objects.empty());
            rlt::rendering::Object object;
            rlt::rendering::ObjectAssembly assembly;
            EXPECT_TRUE(load(device, object, path));
            EXPECT_TRUE(load(device, assembly, path));
        }
    }
}

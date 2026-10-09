#include <rl_tools/operations/cpu.h>
#include <rl_tools/rendering/datasets/glb/operations_cpu.h>
#include <gtest/gtest.h>
#include <rl_tools/rendering/segmentation_cpu.h>
#include <rl_tools/rendering/datasets/procthor/operations_cpu.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <set>

namespace {
    namespace rlt = rl_tools;
    using Json = nlohmann::json;
    using Environment = rlt::rendering::Scene::Environment;
    using Roots = rlt::Mode<rlt::mode::rendering::SceneRoots<>>;
    using rlt::add;
    using rlt::load;
    using rlt::enumerate;
    using rlt::segmentation_taxonomy;
    using rlt::validate_semantic_segmentation;
    using rlt::assign_segmentation_class;
    using rlt::remap_segmentation_classes;
    using rlt::segmentation_objects;

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

        std::string write_glb(std::optional<Json> environment, const Json& classes = Json(), bool single_root = false, const Json& overrides = Json()){
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
            if(!overrides.is_null()){
                document = document.patch(overrides);
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

    TEST_F(GlbEnvironment, welding_ignores_empty_roots_and_keeps_descendant_geometry){
        for(size_t empty_root : {0u, 1u}){
            Json patch = Json::array({{{"op", "remove"}, {"path", "/nodes/" + std::to_string(empty_root) + "/mesh"}},
                                     {{"op", "add"}, {"path", "/nodes/-"}, {"value", {{"mesh", 0}}}},
                                     {{"op", "remove"}, {"path", "/nodes/" + std::to_string(1 - empty_root) + "/mesh"}},
                                     {{"op", "add"}, {"path", "/nodes/" + std::to_string(1 - empty_root) + "/children"}, {"value", {2}}}});
            Json classes = empty_root == 0 ? Json{0, 21} : Json{21, 0};
            const auto path = write_glb(std::nullopt, classes, false, patch);
            rlt::rendering::Object object;
            ASSERT_TRUE(load(device, object, path));
            EXPECT_EQ(object.segmentation_class, 21u);
            EXPECT_TRUE(object.segmentation_class_valid);
            rlt::rendering::ObjectAssembly assembly;
            ASSERT_TRUE(load(device, assembly, path));
            ASSERT_EQ(assembly.parts.size(), 2u);
            EXPECT_TRUE(assembly.objects[empty_root].meshes.empty());
            EXPECT_FALSE(assembly.objects[1 - empty_root].meshes.empty());
        }
    }

    TEST_F(GlbEnvironment, assignment_and_remapping_are_coherent_and_transactional){
        const auto path = write_glb(std::nullopt, Json{21, 162});
        rlt::rendering::Object welded;
        ASSERT_TRUE(load(device, welded, path));
        EXPECT_THROW(remap_segmentation_classes(device, welded, {{0, 21}}, nullptr), std::invalid_argument);
        EXPECT_THROW(assign_segmentation_class(device, welded, 7), std::invalid_argument);
        EXPECT_FALSE(welded.segmentation_class_valid);
        assign_segmentation_class(device, welded, 21);
        EXPECT_TRUE(welded.segmentation_class_valid);
        EXPECT_EQ(welded.segmentation_taxonomy->name, "test");
        EXPECT_NO_THROW(validate_semantic_segmentation(device, segmentation_objects(device, welded)));
        assign_segmentation_class(device, welded, 7, nullptr);
        EXPECT_FALSE(welded.segmentation_taxonomy);
        EXPECT_EQ(welded.segmentation_class, 7u);
        EXPECT_THROW(assign_segmentation_class(device, welded, 0xFFFFFFFFu), std::invalid_argument);
        EXPECT_EQ(welded.segmentation_class, 7u);

        rlt::rendering::Bundle<float> bundle;
        ASSERT_TRUE(load(device, bundle, path, Roots{}));
        const auto source_taxonomy = bundle.scene.objects[0].segmentation_taxonomy;
        auto target = std::make_shared<rlt::rendering::SegmentationTaxonomy>();
        target->name = "target"; target->classes = {{0, "Unknown"}, {7, "Furniture"}, {8, "Structure"}};
        EXPECT_THROW(remap_segmentation_classes(device, bundle.scene, {{21, 7}}, target), std::invalid_argument);
        EXPECT_THROW(remap_segmentation_classes(device, bundle.scene, {{21, 7}, {162, 9}}, target), std::invalid_argument);
        EXPECT_EQ(bundle.scene.objects[0].segmentation_class, 21u);
        EXPECT_EQ(bundle.scene.objects[0].segmentation_taxonomy, source_taxonomy);
        assign_segmentation_class(device, bundle.scene, 0, 162);
        assign_segmentation_class(device, bundle.scene, 0, 21);
        remap_segmentation_classes(device, bundle.scene, {{21, 7}, {162, 8}}, target);
        EXPECT_EQ(bundle.scene.objects[0].segmentation_class, 7u);
        EXPECT_EQ(bundle.scene.objects[1].segmentation_class, 8u);
        EXPECT_EQ(bundle.scene.objects[1].segmentation_taxonomy, target);
        EXPECT_EQ(source_taxonomy->classes.at(21), "Chair");

        rlt::rendering::ObjectAssembly assembly;
        ASSERT_TRUE(load(device, assembly, path));
        assign_segmentation_class(device, assembly, 0, 162);
        rlt::rendering::AssetPool pool;
        pool.assemblies = {assembly, assembly};
        assign_segmentation_class(device, pool, 0, 0, 21);
        remap_segmentation_classes(device, pool, {{21, 7}, {162, 8}}, target);
        EXPECT_EQ(pool.assemblies[0].objects[0].segmentation_class, 7u);
        EXPECT_EQ(pool.assemblies[1].objects[0].segmentation_class, 8u);
        EXPECT_EQ(segmentation_taxonomy(device, pool), target);
        EXPECT_EQ(assembly.objects[0].segmentation_class, 162u);
        remap_segmentation_classes(device, assembly, {{162, 2147483648u}}, nullptr);
        EXPECT_EQ(assembly.objects[1].segmentation_class, 2147483648u);
        EXPECT_FALSE(segmentation_taxonomy(device, assembly));
    }

    TEST_F(GlbEnvironment, numeric_label_boundaries){
        for(const auto& value : {Json(2147483647u), Json(2147483648u), Json(4294967294u), Json(2147483648.0), Json(4294967294.0)}){
            const auto patch = Json::array({{{"op", "add"}, {"path", "/scenes/1/extras/rl_tools/segmentation/classes/Large"}, {"value", value}}});
            const auto path = write_glb(std::nullopt, Json{value, value}, false, patch);
            rlt::rendering::Object object;
            ASSERT_TRUE(load(device, object, path));
            EXPECT_EQ(object.segmentation_class, value.get<uint32_t>());
            EXPECT_EQ(object.segmentation_taxonomy->classes.at(value.get<uint32_t>()), "Large");
        }
        for(const auto& value : {Json(-1), Json(0xFFFFFFFFu), Json(4294967296ULL), Json(UINT64_MAX), Json(21.5), Json("21"), Json(true)}){
            const auto patch = Json::array({{{"op", "add"}, {"path", "/scenes/1/extras/rl_tools/segmentation/classes/Large"}, {"value", value}}});
            const auto path = write_glb(std::nullopt, Json{0, 0}, false, patch);
            rlt::rendering::Object object;
            EXPECT_FALSE(load(device, object, path));
        }
    }

    TEST_F(GlbEnvironment, rejects_invalid_labels_and_taxonomy_conflicts){
        for(const auto& invalid : {Json(-1), Json(999), Json(0xFFFFFFFFu), Json(21.5), Json(4294967296ULL), Json(UINT64_MAX), Json("21")}){
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

    TEST_F(GlbEnvironment, procthor_train_200_identity){
        using DATASET = rlt::rendering::datasets::procthor::Train200;
        DATASET::Corpus corpus{{"discarded"}};
        enumerate(device, DATASET{}, corpus);
        ASSERT_EQ(corpus.references.size(), 200u);
        std::set<std::string> references;
        std::string identity;
        for(size_t scene_i = 0; scene_i < corpus.references.size(); scene_i++){
            const auto& scene = DATASET::SCENES[scene_i];
            EXPECT_EQ(scene.id, "ProcTHOR-Train-" + std::to_string(scene_i));
            EXPECT_EQ(corpus.references[scene_i], scene.reference);
            const std::string reference = scene.reference;
            ASSERT_EQ(reference.substr(0, 6), "conta:");
            std::string hash = reference.substr(6);
            ASSERT_TRUE(conta::detail::normalize_hash(hash));
            EXPECT_TRUE(references.insert(reference).second);
            identity += scene.id;
            identity += '\0';
            identity += reference + '\n';
        }
        EXPECT_EQ(conta::detail::sha256_text(identity), "9115ad15f9d9c9d27b52943fd8d8c273adbe1e1ceafc6f1c79d716dc03c04989");
    }

    TEST_F(GlbEnvironment, catalog_loader_forwards_roots_mode){
        using DATASET = rlt::rendering::datasets::procthor::Train200;
        const auto path = write_glb(authored, Json{21, 162});
        DATASET::Corpus corpus{{path}};
        rlt::rendering::Bundle<float> roots;
        ASSERT_TRUE((load<rlt::rendering::Low, false>(device, DATASET{}, corpus, 0, roots, Roots{})));
        ASSERT_EQ(roots.scene.instances.size(), 2u);
        EXPECT_EQ(roots.scene.objects[1].segmentation_class, 162u);
        expect_authored(roots.scene.environment);
    }

#ifdef RL_TOOLS_TEST_PROCTHOR_CONVERSION
    TEST_F(GlbEnvironment, converter_large_ids_round_trip){
        const auto source = write_glb(std::nullopt);
        std::filesystem::create_directories(directory / "assets");
        std::filesystem::create_directories(directory / "configs");
        std::filesystem::copy_file(source, directory / "assets/stage.glb");
        std::filesystem::copy_file(source, directory / "assets/object.glb");
        const auto input = directory / "configs/test.scene_instance.json";
        std::ofstream(input) << Json{{"stage_instance", {{"template_name", "stage"}}}, {"default_lighting", "no_lights"}, {"object_instances", {{{"template_name", "object"}}}}};
        for(uint32_t id : {2147483647u, 2147483648u, 4294967294u}){
            std::ofstream(directory / "configs/object_semantic_id_mapping.json") << Json{{"Undefined", 0}, {"Large", id}};
            std::ofstream(directory / "configs/object.object_config.json") << Json{{"semantic_id", id}};
            rlt::rendering::datasets::procthor::conversion::Options options;
            options.scene_instance_path = input.string();
            options.output_glb_path = (directory / "converted.glb").string();
            std::string error;
            ASSERT_TRUE(rlt::rendering::datasets::procthor::conversion::scene_instance_to_glb(options, error)) << error;
            rlt::rendering::Bundle<float> bundle;
            ASSERT_TRUE(load(device, bundle, options.output_glb_path, Roots{}));
            EXPECT_EQ(bundle.scene.objects.back().segmentation_class, id);
            EXPECT_EQ(segmentation_taxonomy(device, bundle.scene)->classes.at(id), "Large");
        }
    }

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

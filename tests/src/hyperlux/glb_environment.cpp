#include <rl_tools/operations/cpu.h>
#include <rl_tools/hyperlux/datasets/glb/operations_cpu.h>
#include <gtest/gtest.h>
#include <rl_tools/hyperlux/segmentation_cpu.h>
#include <rl_tools/hyperlux/datasets/procthor/operations_cpu.h>
#include <rl_tools/hyperlux/datasets/hssd/operations_cpu.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>

namespace {
    namespace rlt = rl_tools;
    using Json = nlohmann::json;
    using Environment = rlt::hyperlux::Scene::Environment;
    using Roots = rlt::Mode<rlt::mode::hyperlux::SceneRoots<>>;
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

    TEST_F(GlbEnvironment, shared_textures_survive_asset_copies){
        const auto path = write_glb(std::nullopt, Json(), false, Json{
            {{"op", "add"}, {"path", "/images"}, {"value", {{{"uri", "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAYAAAD0In+KAAAAD0lEQVR4nGP4z8DwHwgbABB5A359Y87XAAAAAElFTkSuQmCC"}}}}},
            {{"op", "add"}, {"path", "/textures"}, {"value", {{{"source", 0}}}}},
            {{"op", "add"}, {"path", "/materials"}, {"value", {{
                {"pbrMetallicRoughness", {{"baseColorTexture", {{"index", 0}}}}},
                {"normalTexture", {{"index", 0}}},
                {"emissiveTexture", {{"index", 0}}},
                {"occlusionTexture", {{"index", 0}}}
            }}}},
            {{"op", "add"}, {"path", "/meshes/0/primitives/0/material"}, {"value", 0}}
        });
        rlt::hyperlux::ObjectAssembly assembly;
        ASSERT_TRUE(load(device, assembly, path));
        ASSERT_EQ(assembly.objects.size(), 2u);
        const auto& first = assembly.objects[0].meshes[0];
        const auto& second = assembly.objects[1].meshes[0];
        ASSERT_TRUE(first.texture.present());
        EXPECT_EQ(first.texture.width, 2);
        EXPECT_EQ(first.texture.height, 1);
        EXPECT_EQ(first.texture.pixels, second.texture.pixels);
        EXPECT_EQ(first.texture.pixels, first.normal_map.pixels);
        EXPECT_EQ(first.texture.pixels, first.emissive_map.pixels);
        EXPECT_EQ(first.texture.pixels, first.occlusion_map.pixels);
        rlt::hyperlux::AssetPool pool;
        pool.assemblies.push_back(assembly);
        EXPECT_EQ(first.texture.pixels, pool.assemblies[0].objects[0].meshes[0].texture.pixels);
        assembly = {};
        const auto& retained = pool.assemblies[0].objects[0].meshes[0].texture;
        ASSERT_TRUE(retained.present());
        EXPECT_EQ(*retained.pixels, (std::vector<uint8_t>{255, 0, 0, 255, 0, 255, 0, 128}));
    }

    TEST_F(GlbEnvironment, semantic_roots_and_welded_rejection){
        const auto path = write_glb(authored, Json{21, 162});
        rlt::hyperlux::Bundle<float> roots, welded;
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
        std::vector<const rlt::hyperlux::Object*> objects = {&welded.scene.objects[0]};
        EXPECT_THROW(validate_semantic_segmentation(device, objects), std::invalid_argument);
        rlt::hyperlux::ObjectAssembly assembly;
        ASSERT_TRUE(load(device, assembly, path));
        rlt::hyperlux::AssetPool pool;
        pool.assemblies.push_back(assembly);
        EXPECT_EQ(segmentation_taxonomy(device, pool.assemblies[0])->classes, taxonomy->classes);
        EXPECT_EQ(pool.assemblies[0].objects[1].segmentation_class, 162u);
    }

    TEST_F(GlbEnvironment, one_root_repeated_classes_and_unknown){
        for(const auto& classes : {Json{21, 21}, Json{0, 0}}){
            const auto path = write_glb(std::nullopt, classes);
            rlt::hyperlux::Object object;
            ASSERT_TRUE(load(device, object, path));
            EXPECT_TRUE(object.segmentation_class_valid);
            EXPECT_EQ(object.segmentation_class, classes[0]);
        }
        const auto path = write_glb(std::nullopt, Json{21}, true);
        rlt::hyperlux::Bundle<float> roots;
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
            rlt::hyperlux::Object object;
            ASSERT_TRUE(load(device, object, path));
            EXPECT_EQ(object.segmentation_class, 21u);
            EXPECT_TRUE(object.segmentation_class_valid);
            rlt::hyperlux::ObjectAssembly assembly;
            ASSERT_TRUE(load(device, assembly, path));
            ASSERT_EQ(assembly.parts.size(), 2u);
            EXPECT_TRUE(assembly.objects[empty_root].meshes.empty());
            EXPECT_FALSE(assembly.objects[1 - empty_root].meshes.empty());
        }
    }

    TEST_F(GlbEnvironment, assignment_and_remapping_are_coherent_and_transactional){
        const auto path = write_glb(std::nullopt, Json{21, 162});
        rlt::hyperlux::Object welded;
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

        rlt::hyperlux::Bundle<float> bundle;
        ASSERT_TRUE(load(device, bundle, path, Roots{}));
        const auto source_taxonomy = bundle.scene.objects[0].segmentation_taxonomy;
        auto target = std::make_shared<rlt::hyperlux::SegmentationTaxonomy>();
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

        rlt::hyperlux::ObjectAssembly assembly;
        ASSERT_TRUE(load(device, assembly, path));
        assign_segmentation_class(device, assembly, 0, 162);
        rlt::hyperlux::AssetPool pool;
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
        for(const auto& value : {Json(2147483647u), Json(2147483648u), Json(4294967039u), Json(2147483648.0), Json(4294967039.0)}){
            const auto patch = Json::array({{{"op", "add"}, {"path", "/scenes/1/extras/rl_tools/segmentation/classes/Large"}, {"value", value}}});
            const auto path = write_glb(std::nullopt, Json{value, value}, false, patch);
            rlt::hyperlux::Object object;
            ASSERT_TRUE(load(device, object, path));
            EXPECT_EQ(object.segmentation_class, value.get<uint32_t>());
            EXPECT_EQ(object.segmentation_taxonomy->classes.at(value.get<uint32_t>()), "Large");
        }
        for(const auto& value : {Json(-1), Json(4294967040u), Json(4294967294u), Json(0xFFFFFFFFu), Json(4294967296ULL), Json(UINT64_MAX), Json(21.5), Json("21"), Json(true)}){
            const auto patch = Json::array({{{"op", "add"}, {"path", "/scenes/1/extras/rl_tools/segmentation/classes/Large"}, {"value", value}}});
            const auto path = write_glb(std::nullopt, Json{0, 0}, false, patch);
            rlt::hyperlux::Object object;
            EXPECT_FALSE(load(device, object, path));
        }
    }

    TEST_F(GlbEnvironment, rejects_invalid_labels_and_taxonomy_conflicts){
        for(const auto& invalid : {Json(-1), Json(999), Json(0xFFFFFFFFu), Json(21.5), Json(4294967296ULL), Json(UINT64_MAX), Json("21")}){
            const auto path = write_glb(std::nullopt, Json{invalid, 21});
            rlt::hyperlux::Bundle<float> roots;
            EXPECT_FALSE(load(device, roots, path, Roots{}));
        }
        rlt::hyperlux::Object first, second;
        auto a = std::make_shared<rlt::hyperlux::SegmentationTaxonomy>();
        a->name = "a"; a->classes = {{0, "Unknown"}};
        auto b = std::make_shared<rlt::hyperlux::SegmentationTaxonomy>(*a);
        b->name = "b";
        first.segmentation_taxonomy = a; second.segmentation_taxonomy = b;
        std::vector<const rlt::hyperlux::Object*> objects = {&first, &second};
        EXPECT_THROW(validate_semantic_segmentation(device, objects), std::invalid_argument);
    }

    TEST_F(GlbEnvironment, glb_dataset_forwards_roots_mode){
        const auto path = write_glb(authored, Json{21, 162});
        rlt::hyperlux::datasets::procthor::GLB dataset;
        rlt::hyperlux::datasets::procthor::Corpus corpus;
        corpus.references.push_back(path);
        rlt::hyperlux::Bundle<float> roots;
        ASSERT_TRUE(load(device, dataset, corpus, 0, roots, Roots{}));
        ASSERT_EQ(roots.scene.instances.size(), 2u);
        EXPECT_EQ(roots.scene.objects[1].segmentation_class, 162u);
    }

    TEST_F(GlbEnvironment, procthor_train_200_identity){
        using DATASET = rlt::hyperlux::datasets::procthor::Train200;
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
        using DATASET = rlt::hyperlux::datasets::procthor::Train200;
        const auto path = write_glb(authored, Json{21, 162});
        DATASET::Corpus corpus{{path}};
        rlt::hyperlux::Bundle<float> roots;
        ASSERT_TRUE((load<rlt::hyperlux::Low, false>(device, DATASET{}, corpus, 0, roots, Roots{})));
        ASSERT_EQ(roots.scene.instances.size(), 2u);
        EXPECT_EQ(roots.scene.objects[1].segmentation_class, 162u);
        expect_authored(roots.scene.environment);
    }

    TEST_F(GlbEnvironment, hssd_catalogs_identity){
        std::set<std::string> ids, references;
        const auto identity = [&](auto dataset){
            using DATASET = decltype(dataset);
            typename DATASET::Corpus corpus{{"discarded"}};
            enumerate(device, dataset, corpus);
            EXPECT_EQ(corpus.references.size(), DATASET::NUM_SCENES);
            std::string text;
            for(size_t scene_i = 0; scene_i < DATASET::NUM_SCENES; scene_i++){
                const auto& scene = DATASET::SCENES[scene_i];
                const std::string id = scene.id, reference = scene.reference;
                EXPECT_EQ(corpus.references[scene_i], reference);
                EXPECT_TRUE(!id.empty() && id.find_first_not_of("0123456789_") == std::string::npos && id.front() != '_' && id.back() != '_') << id;
                EXPECT_TRUE(ids.insert(id).second) << id;
                EXPECT_EQ(reference.substr(0, 6), "conta:");
                std::string hash = reference.substr(6);
                EXPECT_TRUE(conta::detail::normalize_hash(hash)) << reference;
                EXPECT_TRUE(references.insert(reference).second) << reference;
                text += id;
                text += '\0';
                text += reference + '\n';
            }
            return conta::detail::sha256_text(text);
        };
        EXPECT_EQ(rlt::hyperlux::datasets::hssd::Train125::NUM_SCENES, 125u);
        EXPECT_EQ(rlt::hyperlux::datasets::hssd::Val43::NUM_SCENES, 43u);
        EXPECT_EQ(identity(rlt::hyperlux::datasets::hssd::Train125{}), "7cd6dbc35e1b4dcad093b72b10a4078b4b658948e517166253e99da57f1c8159");
        EXPECT_EQ(identity(rlt::hyperlux::datasets::hssd::Val43{}), "75581ea7f0a030eafdab0cbb74206bbfd77543f0c88fee9ceb93fb610119ef0c");
    }

    TEST_F(GlbEnvironment, hssd_catalog_loaders_forward_roots_mode){
        const auto path = write_glb(authored, Json{21, 162});
        const auto expect_roots = [&](auto dataset){
            typename decltype(dataset)::Corpus corpus{{path}};
            rlt::hyperlux::Bundle<float> roots;
            ASSERT_TRUE((load<rlt::hyperlux::Low, false>(device, dataset, corpus, 0, roots, Roots{})));
            ASSERT_EQ(roots.scene.instances.size(), 2u);
            EXPECT_EQ(roots.scene.objects[1].segmentation_class, 162u);
        };
        expect_roots(rlt::hyperlux::datasets::hssd::Train125{});
        expect_roots(rlt::hyperlux::datasets::hssd::Val43{});
    }

    TEST_F(GlbEnvironment, hssd_catalog_scene_carries_labels){
        if(std::getenv("CONTA_ROOT") == nullptr){
            GTEST_SKIP() << "Set CONTA_ROOT to a local store containing the HSSD scenes";
        }
        using DATASET = rlt::hyperlux::datasets::hssd::Train125;
        const std::string reference = DATASET::SCENES[63].reference;
        ASSERT_EQ(std::string(DATASET::SCENES[63].id), "107734119_175999932");
        std::string path, error;
        if(!conta::resolve(reference.substr(6), path, error)){
            GTEST_SKIP() << error;
        }
        DATASET::Corpus corpus;
        enumerate(device, DATASET{}, corpus);
        rlt::hyperlux::Bundle<float> bundle;
        ASSERT_TRUE((load<rlt::hyperlux::Low, false>(device, DATASET{}, corpus, 63, bundle, Roots{})));
        const auto taxonomy = segmentation_taxonomy(device, bundle.scene);
        ASSERT_TRUE(taxonomy);
        EXPECT_EQ(taxonomy->name, DATASET::SEGMENTATION_TAXONOMY);
        EXPECT_EQ(taxonomy->classes.size(), 406u);
        size_t labelled = 0;
        for(const auto& object : bundle.scene.objects){
            labelled += object.segmentation_class != 0;
        }
        EXPECT_GT(labelled, 0u);
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
        for(uint32_t id : {2147483647u, 2147483648u, 4294967039u}){
            std::ofstream(directory / "configs/object_semantic_id_mapping.json") << Json{{"Undefined", 0}, {"Large", id}};
            std::ofstream(directory / "configs/object.object_config.json") << Json{{"semantic_id", id}};
            rlt::hyperlux::datasets::procthor::conversion::Options options;
            options.scene_instance_path = input.string();
            options.output_glb_path = (directory / "converted.glb").string();
            std::string error;
            ASSERT_TRUE(rlt::hyperlux::datasets::procthor::conversion::scene_instance_to_glb(options, error)) << error;
            rlt::hyperlux::Bundle<float> bundle;
            ASSERT_TRUE(load(device, bundle, options.output_glb_path, Roots{}));
            EXPECT_EQ(bundle.scene.objects.back().segmentation_class, id);
            EXPECT_EQ(segmentation_taxonomy(device, bundle.scene)->classes.at(id), "Large");
        }
    }

    TEST_F(GlbEnvironment, converter_hssd_round_trip){
        const auto source = write_glb(std::nullopt);
        for(const auto* subdirectory : {"stages", "objects", "metadata", "semantics/scenes", "scenes"}){
            std::filesystem::create_directories(directory / subdirectory);
        }
        std::filesystem::copy_file(source, directory / "stages/stage.glb");
        std::filesystem::copy_file(source, directory / "objects/object.glb");
        std::ofstream(directory / "objects/object.object_config.json") << Json{{"semantic_id", 7}};
        std::ofstream(directory / "semantics/hssd-hab_semantic_lexicon.json") << Json{{"classes", {{{"id", 0}, {"name", "unknown"}}, {{"id", 7}, {"name", "chair"}}, {{"id", 300}, {"name", "table"}}}}};
        std::ofstream(directory / "metadata/hssd_obj_semantics_condensed.csv") << "Object Hash,\"Is Pickable:\nsmall?\",\"Semantic Category:\nCONDENSED\",\"Semantic Category:\nThe primary\"\nobject,No,table,dining_table\n";
        std::ofstream(directory / "semantics/objects.csv") << "id,wnsynsetkey,main_category,super_category\n";
        const Json room = {{"label", "kitchen"}, {"floor_height", 0}, {"extrusion_height", 2.8}, {"poly_loop", {{0, 0, 0}, {3, 0, 0}, {0, 0, 3}}}};
        std::ofstream(directory / "semantics/scenes/test.semantic_config.json") << Json{{"region_annotations", Json::array({room})}};
        const auto input = directory / "scenes/test.scene_instance.json";
        std::ofstream(input) << Json{{"stage_instance", {{"template_name", "stages/stage"}}}, {"default_lighting", "no_lights"}, {"object_instances", {{{"template_name", "objects/object"}}}}};
        rlt::hyperlux::datasets::procthor::conversion::Options options;
        options.scene_instance_path = input.string();
        options.output_glb_path = (directory / "converted.glb").string();
        options.dataset = rlt::hyperlux::datasets::procthor::conversion::Dataset::HSSD;
        std::string error;
        ASSERT_TRUE(rlt::hyperlux::datasets::procthor::conversion::scene_instance_to_glb(options, error)) << error;
        rlt::hyperlux::Bundle<float> bundle;
        ASSERT_TRUE(load(device, bundle, options.output_glb_path, Roots{}));
        EXPECT_TRUE(bundle.metadata.regions.annotated);
        ASSERT_EQ(bundle.metadata.regions.volumes.size(), 1u);
        EXPECT_TRUE(bundle.metadata.regions.volumes[0].indoor);
        ASSERT_EQ(bundle.scene.objects.size(), 3u);
        EXPECT_EQ(bundle.scene.objects[0].segmentation_class, 0u);
        EXPECT_EQ(bundle.scene.objects[1].segmentation_class, 0u);
        EXPECT_EQ(bundle.scene.objects[2].segmentation_class, 300u);
        const auto taxonomy = segmentation_taxonomy(device, bundle.scene);
        ASSERT_TRUE(taxonomy);
        EXPECT_EQ(taxonomy->name, "hssd-hab");
        EXPECT_EQ(taxonomy->classes, (std::map<uint32_t, std::string>{{0, "unknown"}, {7, "chair"}, {300, "table"}}));
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
        rlt::hyperlux::datasets::procthor::AI2ThorHab dataset;
        dataset.root = directory.string();
        dataset.cache_directory = (directory / "cache").string();
        rlt::hyperlux::datasets::procthor::Corpus corpus;
        corpus.references.push_back(scene.string());
        rlt::hyperlux::Bundle<float> first, second, third;
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
        rlt::hyperlux::Bundle<float> welded, roots;
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
        rlt::hyperlux::ObjectAssembly assembly;
        ASSERT_TRUE(load(device, assembly, path));
        for(size_t part = 0; part < assembly.parts.size(); part++){
            for(int element = 0; element < 12; element++){
                EXPECT_FLOAT_EQ(roots.scene.instances[part].transform[element], assembly.parts[part].transform[element]);
            }
        }
    }

    TEST_F(GlbEnvironment, absent_metadata_keeps_generic_defaults){
        const auto path = write_glb(std::nullopt);
        rlt::hyperlux::Bundle<float> welded, roots;
        ASSERT_TRUE(load(device, welded, path));
        ASSERT_TRUE(load(device, roots, path, Roots{}));
        for(const auto* bundle : {&welded, &roots}){
            EXPECT_FALSE(bundle->metadata.regions.annotated);
            EXPECT_TRUE(bundle->metadata.regions.volumes.empty());
            EXPECT_EQ(bundle->scene.environment.mode, Environment::Mode::DEFAULT);
            for(float ambient : bundle->scene.environment.ambient){
                EXPECT_FLOAT_EQ(ambient, 0.1f);
            }
        }
    }

    TEST_F(GlbEnvironment, region_polygons_preserve_coordinates_heights_and_exterior_labels){
        const Json room = {
            {"label", "kitchen"}, {"floor_height", 3.25}, {"extrusion_height", 2.5},
            {"poly_loop", {{1, 3.25, 2}, {5, 3.25, 2}, {5, 3.25, 4}, {3, 3.25, 4}, {3, 3.25, 6}, {1, 3.25, 6}}},
            {"min_bounds", {1, 3.25, 2}}, {"max_bounds", {5, 2.5, 6}}
        };
        Json regions = Json::array({room});
        for(const auto* label : {"outdoor", "porch/terrace/deck/driveway", "balcony"}){
            regions.push_back(room);
            regions.back()["label"] = label;
        }
        const auto path = write_glb(authored, Json(), false, Json::array({
            {{"op", "add"}, {"path", "/scenes/0/extras/hssd"}, {"value", {{"region_annotations", "invalid inactive scene"}}}},
            {{"op", "add"}, {"path", "/scenes/1/extras/hssd"}, {"value", {{"region_annotations", regions}}}}
        }));
        rlt::hyperlux::Bundle<float> welded, roots;
        ASSERT_TRUE(load(device, welded, path));
        ASSERT_TRUE(load(device, roots, path, Roots{}));
        for(const auto* bundle : {&welded, &roots}){
            const auto& parsed = bundle->metadata.regions;
            EXPECT_TRUE(parsed.annotated);
            ASSERT_EQ(parsed.volumes.size(), 4u);
            EXPECT_TRUE(parsed.volumes[0].indoor);
            for(size_t i = 0; i < parsed.volumes.size(); i++){
                const auto& region = parsed.volumes[i];
                EXPECT_EQ(region.indoor, i == 0);
                ASSERT_EQ(region.polygon.size(), 6u);
                EXPECT_EQ(region.polygon[0], (std::array<float, 2>{1, -2}));
                EXPECT_EQ(region.polygon[4], (std::array<float, 2>{3, -6}));
                EXPECT_FLOAT_EQ(region.min[0], 1);
                EXPECT_FLOAT_EQ(region.min[1], -6);
                EXPECT_FLOAT_EQ(region.min[2], 3.25);
                EXPECT_FLOAT_EQ(region.max[0], 5);
                EXPECT_FLOAT_EQ(region.max[1], -2);
                EXPECT_FLOAT_EQ(region.max[2], 5.75);
            }
        }
    }

    TEST_F(GlbEnvironment, empty_region_annotations_do_not_become_unannotated){
        const auto path = write_glb(authored, Json(), false, Json::array({
            {{"op", "add"}, {"path", "/scenes/1/extras/hssd"}, {"value", {{"region_annotations", Json::array()}}}}
        }));
        rlt::hyperlux::Bundle<float> welded, roots;
        ASSERT_TRUE(load(device, welded, path));
        ASSERT_TRUE(load(device, roots, path, Roots{}));
        for(const auto* bundle : {&welded, &roots}){
            EXPECT_TRUE(bundle->metadata.regions.annotated);
            EXPECT_TRUE(bundle->metadata.regions.volumes.empty());
        }
    }

    TEST_F(GlbEnvironment, malformed_regions_reject_scenes_but_not_asset_import_or_composition){
        const Json room = {{"label", "kitchen"}, {"floor_height", 0}, {"extrusion_height", 2.8}, {"poly_loop", {{0, 0, 0}, {3, 0, 0}, {0, 0, 3}}}};
        std::vector<Json> invalid{nullptr, 1, Json::object(), {{"region_annotations", nullptr}}, {{"region_annotations", "invalid"}}};
        for(const auto& patch : Json::array({
            Json::array({{{"op", "remove"}, {"path", "/label"}}}),
            Json::array({{{"op", "replace"}, {"path", "/label"}, {"value", ""}}}),
            Json::array({{{"op", "remove"}, {"path", "/floor_height"}}}),
            Json::array({{{"op", "replace"}, {"path", "/extrusion_height"}, {"value", -1}}}),
            Json::array({{{"op", "replace"}, {"path", "/floor_height"}, {"value", "0"}}}),
            Json::array({{{"op", "replace"}, {"path", "/poly_loop"}, {"value", {{0, 0, 0}, {1, 0, 1}}}}}),
            Json::array({{{"op", "replace"}, {"path", "/poly_loop/0"}, {"value", {0, 0}}}}),
            Json::array({{{"op", "replace"}, {"path", "/poly_loop/0/0"}, {"value", 1e100}}}),
            Json::array({{{"op", "replace"}, {"path", "/poly_loop"}, {"value", {{0, 0, 0}, {1, 0, 1}, {2, 0, 2}}}}})
        })){
            invalid.push_back({{"region_annotations", Json::array({room.patch(patch)})}});
        }
        for(const auto& hssd : invalid){
            SCOPED_TRACE(hssd.dump());
            const auto path = write_glb(authored, Json(), false, Json::array({
                {{"op", "add"}, {"path", "/scenes/1/extras/hssd"}, {"value", hssd}}
            }));
            rlt::hyperlux::Bundle<float> welded, roots, composed;
            EXPECT_FALSE(load(device, welded, path));
            EXPECT_FALSE(load(device, roots, path, Roots{}));
            EXPECT_TRUE(welded.scene.objects.empty());
            EXPECT_TRUE(roots.scene.objects.empty());
            rlt::hyperlux::Object object;
            rlt::hyperlux::ObjectAssembly assembly;
            EXPECT_TRUE(load(device, object, path));
            EXPECT_TRUE(load(device, assembly, path));
            composed.metadata.regions.annotated = true;
            composed.metadata.regions.volumes.resize(1);
            composed.metadata.regions.volumes[0].min[0] = 17;
            EXPECT_TRUE(add(device, composed, path));
            EXPECT_TRUE(composed.metadata.regions.annotated);
            ASSERT_EQ(composed.metadata.regions.volumes.size(), 1u);
            EXPECT_FLOAT_EQ(composed.metadata.regions.volumes[0].min[0], 17);
        }
    }

    TEST_F(GlbEnvironment, composition_preserves_parent_environment){
        const auto path = write_glb(authored);
        rlt::hyperlux::Bundle<float> bundle;
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
            rlt::hyperlux::Bundle<float> welded, roots;
            EXPECT_FALSE(load(device, welded, path));
            EXPECT_FALSE(load(device, roots, path, Roots{}));
            EXPECT_TRUE(welded.scene.objects.empty());
            EXPECT_TRUE(roots.scene.objects.empty());
            rlt::hyperlux::Object object;
            rlt::hyperlux::ObjectAssembly assembly;
            EXPECT_TRUE(load(device, object, path));
            EXPECT_TRUE(load(device, assembly, path));
        }
    }
}

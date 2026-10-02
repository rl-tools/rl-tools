#include <rl_tools/operations/cpu.h>
#include <rl_tools/rendering/datasets/glb/operations_cpu.h>
#include <gtest/gtest.h>

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

        std::string write_glb(std::optional<Json> environment){
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

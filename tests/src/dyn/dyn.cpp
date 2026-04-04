#include <rl_tools/operations/cpu.h>
#include <rl_tools/persist/backends/tar/operations_cpu.h>
#include <rl_tools/persist/backends/tar/operations_generic.h>

#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/max_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/avg_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/flatten/operations_generic.h>
#include <rl_tools/nn/layers/resnet_block/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/resnet/resnet.h>

#include <rl_tools/nn/parameters/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/gru/persist.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/max_pool2d/persist.h>
#include <rl_tools/nn/layers/avg_pool2d/persist.h>
#include <rl_tools/nn/layers/flatten/persist.h>
#include <rl_tools/nn/layers/resnet_block/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>

#include <rl_tools/dyn/persist.h>

#define RL_TOOLS_STRINGIZE(x) #x
#define RL_TOOLS_MACRO_TO_STR(macro) RL_TOOLS_STRINGIZE(macro)

namespace rlt = rl_tools;

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <cmath>

using DEVICE = rl_tools::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using TI = typename DEVICE::index_t;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;

#include <gtest/gtest.h>

namespace helpers{
    void setup_buffer(rlt::dyn::Buffer<TI>& buf, rlt::dyn::Layer<TI>& layer, const rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>>& input){
        rlt::dyn::propagate_shapes(layer, input.shape, input.rank, input.size);
        buf.layer = &layer;
    }
    std::vector<char> load_tar(const std::string& path){
        std::ifstream archive(path, std::ios::binary);
        std::vector<char> data((std::istreambuf_iterator<char>(archive)), std::istreambuf_iterator<char>());
        archive.close();
        return data;
    }
}

TEST(TEST_DYN, dense_layer){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);

    constexpr TI INPUT_DIM = 15;
    constexpr TI OUTPUT_DIM = 10;
    using CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, OUTPUT_DIM, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CAPABILITY = rlt::nn::capability::Forward<>;
    using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, 3, INPUT_DIM>;
    using SPEC = rlt::nn::layers::dense::Specification<CONFIG, CAPABILITY, INPUT_SHAPE>;
    rlt::nn::layers::dense::LayerForward<SPEC> layer;
    rlt::malloc(device, layer);
    rlt::init_weights(device, layer, rng);

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename SPEC::OUTPUT_SHAPE>> output_static;
    rlt::malloc(device, input);
    rlt::malloc(device, output_static);
    rlt::randn(device, input, rng);

    rlt::nn::layers::dense::Buffer buffer_static;
    rlt::evaluate(device, layer, input, output_static, buffer_static, rng);

    // Save to TAR
    std::string data_path = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/test_dyn_dense.tar";
    rlt::persist::backends::tar::Writer writer;
    DEVICE finalize_device;
    rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(writer)>> writer_group{"", &writer};
    auto layer_group = rlt::create_group(device, writer_group, "layer");
    rlt::save(device, layer, layer_group);
    rlt::persist::backends::tar::finalize(device, writer);
    std::ofstream archive(data_path, std::ios::binary);
    archive.write(writer.buffer.data(), writer.buffer.size());
    archive.close();

    // Load into dyn
    auto tar_data = helpers::load_tar(data_path);
    rlt::persist::backends::tar::ReaderGroup<rlt::persist::backends::tar::ReaderGroupSpecification<TI>> reader_group{"", tar_data.data(), static_cast<TI>(tar_data.size())};
    auto dyn_layer_group = rlt::get_group(device, reader_group, "layer");

    rlt::dyn::Layer<TI> dyn_layer;
    rlt::load(device, dyn_layer, dyn_layer_group);
    ASSERT_EQ(dyn_layer.type, rlt::dyn::LayerType::DENSE);

    // Prepare dyn input/output
    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> dyn_input, dyn_output;
    TI input_shape[] = {3, INPUT_DIM};
    rlt::dyn::set_shape(dyn_input, (TI)2, input_shape);
    dyn_input.type = rlt::dyn::Type::FLOAT32;
    rlt::malloc(device, dyn_input);
    // Copy input data
    auto input_mat = rlt::matrix_view(device, input);
    for(TI i = 0; i < 3; i++){
        for(TI j = 0; j < INPUT_DIM; j++){
            rlt::dyn::set(device, dyn_input, i * INPUT_DIM + j, rlt::get(input_mat, i, j));
        }
    }

    TI output_shape[] = {3, OUTPUT_DIM};
    rlt::dyn::set_shape(dyn_output, (TI)2, output_shape);
    dyn_output.type = rlt::dyn::Type::FLOAT32;
    rlt::malloc(device, dyn_output);

    rlt::dyn::Buffer<TI> dyn_buffer;
    helpers::setup_buffer(dyn_buffer, dyn_layer, dyn_input);
    rlt::malloc(device, dyn_buffer);

    ASSERT_TRUE(rlt::evaluate(device, dyn_layer, dyn_input, dyn_output, dyn_buffer));

    // Compare outputs
    auto output_mat = rlt::matrix_view(device, output_static);
    T max_diff = 0;
    for(TI i = 0; i < 3; i++){
        for(TI j = 0; j < OUTPUT_DIM; j++){
            T static_val = rlt::get(output_mat, i, j);
            T dyn_val = rlt::dyn::get(device, dyn_output, i * OUTPUT_DIM + j);
            T diff = std::abs(static_val - dyn_val);
            if(diff > max_diff) max_diff = diff;
        }
    }
    std::cout << "Dense max diff: " << max_diff << std::endl;
    ASSERT_NEAR(max_diff, 0, 1e-5);

    rlt::free(device, dyn_input);
    rlt::free(device, dyn_output);
    rlt::free(device, dyn_buffer);
    rlt::free(device, dyn_layer);
    rlt::free(device, layer);
    rlt::free(device, input);
    rlt::free(device, output_static);
}

TEST(TEST_DYN, gru_single_step){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 77);
    constexpr TI GI = 4, GH = 3, GB = 1;
    using GC = rlt::nn::layers::gru::Configuration<TYPE_POLICY, TI, GH>;
    using GS = rlt::nn::layers::gru::Specification<GC, rlt::nn::capability::Forward<>, rlt::tensor::Shape<TI, 1, GB, GI>>;
    rlt::nn::layers::gru::LayerForward<GS> layer;
    rlt::malloc(device, layer);
    rlt::init_weights(device, layer, rng);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, GB, GI>>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, GB, GH>>> output_static;
    rlt::malloc(device, input); rlt::malloc(device, output_static);
    rlt::randn(device, input, rng);
    typename rlt::nn::layers::gru::LayerForward<GS>::template Buffer<> buf;
    rlt::malloc(device, buf);
    rlt::evaluate(device, layer, input, output_static, buf, rng);
    // Manual dyn computation using layer's weights directly
    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> dyn_in, dyn_out;
    TI is[] = {(TI)1, GB, GI}; rlt::dyn::set_shape(dyn_in, (TI)3, is); dyn_in.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, dyn_in);
    TI os[] = {(TI)1, GB, GH}; rlt::dyn::set_shape(dyn_out, (TI)3, os); dyn_out.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, dyn_out);
    auto im = rlt::matrix_view(device, input);
    for(TI i = 0; i < GB * GI; i++) rlt::dyn::set(device, dyn_in, i, rlt::get(im, 0, i));
    // Build dyn layer manually from static weights
    rlt::dyn::Layer<TI> dl;
    dl.type = rlt::dyn::LayerType::GRU;
    auto* gd = new rlt::dyn::layers::GRU<TI>();
    gd->input_dim = GI; gd->hidden_dim = GH;
    // Copy weights
    auto copy_tensor = [&](auto& src_param, rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>>& dst, TI r, TI c){
        TI sh[] = {r, c}; rlt::dyn::set_shape(dst, (TI)2, sh); dst.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, dst);
        for(TI i = 0; i < r; i++) for(TI j = 0; j < c; j++) rlt::dyn::set(device, dst, i*c+j, (float)rlt::get(device, src_param.parameters, i, j));
    };
    auto copy_tensor_1d = [&](auto& src_param, rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>>& dst, TI n){
        TI sh[] = {n}; rlt::dyn::set_shape(dst, (TI)1, sh); dst.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, dst);
        for(TI i = 0; i < n; i++) rlt::dyn::set(device, dst, i, (float)rlt::get(device, src_param.parameters, i));
    };
    copy_tensor(layer.weights_input, gd->weights_input, 3*GH, GI);
    copy_tensor_1d(layer.biases_input, gd->biases_input, 3*GH);
    copy_tensor(layer.weights_hidden, gd->weights_hidden, 3*GH, GH);
    copy_tensor_1d(layer.biases_hidden, gd->biases_hidden, 3*GH);
    copy_tensor_1d(layer.initial_hidden_state, gd->initial_hidden_state, GH);
    dl.data = gd;
    rlt::dyn::Buffer<TI> db; helpers::setup_buffer(db, dl, dyn_in); rlt::malloc(device, db);
    ASSERT_TRUE(rlt::evaluate(device, dl, dyn_in, dyn_out, db));
    auto om = rlt::matrix_view(device, output_static);
    T md = 0;
    for(TI j = 0; j < GH; j++){
        T sv = rlt::get(om, 0, j);
        T dv = rlt::dyn::get(device, dyn_out, j);
        std::cout << "  h[" << j << "] static=" << sv << " dyn=" << dv << " diff=" << std::abs(sv-dv) << std::endl;
        T d = std::abs(sv-dv); if(d > md) md = d;
    }
    std::cout << "GRU single step max diff: " << md << std::endl;
    ASSERT_NEAR(md, 0, 1e-5);
    rlt::free(device, dyn_in); rlt::free(device, dyn_out); rlt::free(device, db); rlt::free(device, dl);
    rlt::free(device, layer); rlt::free(device, buf); rlt::free(device, input); rlt::free(device, output_static);
}

TEST(TEST_DYN, gru_standalone){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 99);
    constexpr TI GRU_INPUT_DIM = 8;
    constexpr TI HIDDEN_DIM = 12;
    constexpr TI SEQ_LEN = 4;
    constexpr TI GRU_BATCH_SIZE = 2;
    using GRU_CONFIG = rlt::nn::layers::gru::Configuration<TYPE_POLICY, TI, HIDDEN_DIM>;
    using GRU_CAPABILITY = rlt::nn::capability::Forward<>;
    using GRU_INPUT_SHAPE = rlt::tensor::Shape<TI, SEQ_LEN, GRU_BATCH_SIZE, GRU_INPUT_DIM>;
    using GRU_SPEC = rlt::nn::layers::gru::Specification<GRU_CONFIG, GRU_CAPABILITY, GRU_INPUT_SHAPE>;
    rlt::nn::layers::gru::LayerForward<GRU_SPEC> gru_layer;
    rlt::malloc(device, gru_layer);
    rlt::init_weights(device, gru_layer, rng);
    rlt::Tensor<rlt::tensor::Specification<T, TI, GRU_INPUT_SHAPE>> gru_input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename GRU_SPEC::OUTPUT_SHAPE>> gru_output_static;
    rlt::malloc(device, gru_input);
    rlt::malloc(device, gru_output_static);
    rlt::randn(device, gru_input, rng);
    typename rlt::nn::layers::gru::LayerForward<GRU_SPEC>::template Buffer<> gru_buffer_static;
    rlt::malloc(device, gru_buffer_static);
    rlt::evaluate(device, gru_layer, gru_input, gru_output_static, gru_buffer_static, rng);
    std::string data_path = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/test_dyn_gru_standalone.tar";
    rlt::persist::backends::tar::Writer writer;
    rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(writer)>> writer_group{"", &writer};
    auto layer_group = rlt::create_group(device, writer_group, "layer");
    rlt::save(device, gru_layer, layer_group);
    rlt::persist::backends::tar::finalize(device, writer);
    {std::ofstream archive(data_path, std::ios::binary); archive.write(writer.buffer.data(), writer.buffer.size());}
    auto tar_data = helpers::load_tar(data_path);
    rlt::persist::backends::tar::ReaderGroup<rlt::persist::backends::tar::ReaderGroupSpecification<TI>> reader_group{"", tar_data.data(), static_cast<TI>(tar_data.size())};
    auto dyn_layer_group = rlt::get_group(device, reader_group, "layer");
    rlt::dyn::Layer<TI> dyn_gru_layer;
    rlt::load(device, dyn_gru_layer, dyn_layer_group);
    ASSERT_EQ(dyn_gru_layer.type, rlt::dyn::LayerType::GRU);
    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> dyn_gru_input, dyn_gru_output;
    TI gi_shape[] = {SEQ_LEN, GRU_BATCH_SIZE, GRU_INPUT_DIM};
    rlt::dyn::set_shape(dyn_gru_input, (TI)3, gi_shape);
    dyn_gru_input.type = rlt::dyn::Type::FLOAT32;
    rlt::malloc(device, dyn_gru_input);
    auto gru_input_mat = rlt::matrix_view(device, gru_input);
    for(TI i = 0; i < SEQ_LEN * GRU_BATCH_SIZE; i++)
        for(TI j = 0; j < GRU_INPUT_DIM; j++)
            rlt::dyn::set(device, dyn_gru_input, i * GRU_INPUT_DIM + j, rlt::get(gru_input_mat, i, j));
    TI go_shape[] = {SEQ_LEN, GRU_BATCH_SIZE, HIDDEN_DIM};
    rlt::dyn::set_shape(dyn_gru_output, (TI)3, go_shape);
    dyn_gru_output.type = rlt::dyn::Type::FLOAT32;
    rlt::malloc(device, dyn_gru_output);
    rlt::dyn::Buffer<TI> dyn_gru_buffer;
    helpers::setup_buffer(dyn_gru_buffer, dyn_gru_layer, dyn_gru_input);
    rlt::malloc(device, dyn_gru_buffer);
    ASSERT_TRUE(rlt::evaluate(device, dyn_gru_layer, dyn_gru_input, dyn_gru_output, dyn_gru_buffer));
    auto gru_output_mat = rlt::matrix_view(device, gru_output_static);
    T max_diff = 0;
    for(TI i = 0; i < SEQ_LEN * GRU_BATCH_SIZE; i++)
        for(TI j = 0; j < HIDDEN_DIM; j++){
            T diff = std::abs(rlt::get(gru_output_mat, i, j) - rlt::dyn::get(device, dyn_gru_output, i * HIDDEN_DIM + j));
            if(diff > max_diff) max_diff = diff;
        }
    std::cout << "GRU standalone max diff: " << max_diff << std::endl;
    ASSERT_NEAR(max_diff, 0, 1e-5);
    rlt::free(device, dyn_gru_input); rlt::free(device, dyn_gru_output);
    rlt::free(device, dyn_gru_buffer); rlt::free(device, dyn_gru_layer);
    rlt::free(device, gru_layer); rlt::free(device, gru_buffer_static);
    rlt::free(device, gru_input); rlt::free(device, gru_output_static);
}

TEST(TEST_DYN, sequential_dense_dense){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);
    using D1 = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 8, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using D2 = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 5, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using CHAIN = rlt::nn_models::sequential::Module<rlt::nn::layers::dense::BindConfiguration<D1>, rlt::nn_models::sequential::Module<rlt::nn::layers::dense::BindConfiguration<D2>>>;
    using M = rlt::nn_models::sequential::Build<rlt::nn::capability::Forward<>, CHAIN, rlt::tensor::Shape<TI, 3, 6>>;
    M model; M::Buffer<> buf;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename M::INPUT_SHAPE>> in;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename M::OUTPUT_SHAPE>> out;
    rlt::malloc(device, model); rlt::malloc(device, buf); rlt::malloc(device, in); rlt::malloc(device, out);
    rlt::init_weights(device, model, rng); rlt::randn(device, in, rng);
    rlt::evaluate(device, model, in, out, buf, rng);
    std::string dp = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/test_dyn_seq_dd.tar";
    rlt::persist::backends::tar::Writer w;
    rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(w)>> wg{"", &w};
    auto mg = rlt::create_group(device, wg, "m"); rlt::save(device, model, mg);
    rlt::persist::backends::tar::finalize(device, w);
    {std::ofstream a(dp, std::ios::binary); a.write(w.buffer.data(), w.buffer.size());}
    auto td = helpers::load_tar(dp);
    rlt::persist::backends::tar::ReaderGroup<rlt::persist::backends::tar::ReaderGroupSpecification<TI>> rg{"", td.data(), static_cast<TI>(td.size())};
    auto dg = rlt::get_group(device, rg, "m");
    rlt::dyn::Layer<TI> dm; rlt::load(device, dm, dg);
    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> di, d_out;
    TI dis[] = {(TI)3, (TI)6}; rlt::dyn::set_shape(di, (TI)2, dis); di.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, di);
    TI dos[] = {(TI)3, (TI)5}; rlt::dyn::set_shape(d_out, (TI)2, dos); d_out.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, d_out);
    auto im = rlt::matrix_view(device, in);
    for(TI i = 0; i < 3; i++) for(TI j = 0; j < 6; j++) rlt::dyn::set(device, di, i*6+j, rlt::get(im, i, j));
    rlt::dyn::Buffer<TI> db; helpers::setup_buffer(db, dm, di); rlt::malloc(device, db);
    ASSERT_TRUE(rlt::evaluate(device, dm, di, d_out, db));
    auto om = rlt::matrix_view(device, out);
    T md = 0;
    for(TI i = 0; i < 3; i++) for(TI j = 0; j < 5; j++){
        T d = std::abs(rlt::get(om, i, j) - rlt::dyn::get(device, d_out, i*5+j));
        if(d > md) md = d;
    }
    std::cout << "Sequential Dense+Dense max diff: " << md << std::endl;
    ASSERT_NEAR(md, 0, 1e-5);
    rlt::free(device, di); rlt::free(device, d_out); rlt::free(device, db); rlt::free(device, dm);
    rlt::free(device, model); rlt::free(device, buf); rlt::free(device, in); rlt::free(device, out);
}

TEST(TEST_DYN, sequential_dense_gru){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);
    using D1C = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 10, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using GC = rlt::nn::layers::gru::Configuration<TYPE_POLICY, TI, 8>;
    using CHAIN = rlt::nn_models::sequential::Module<rlt::nn::layers::dense::BindConfiguration<D1C>, rlt::nn_models::sequential::Module<rlt::nn::layers::gru::BindConfiguration<GC>>>;
    constexpr TI SL = 4, BS = 2, ID = 10;
    using M = rlt::nn_models::sequential::Build<rlt::nn::capability::Forward<>, CHAIN, rlt::tensor::Shape<TI, SL, BS, ID>>;
    M model; M::Buffer<> buf;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename M::INPUT_SHAPE>> in;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename M::OUTPUT_SHAPE>> out;
    rlt::malloc(device, model); rlt::malloc(device, buf); rlt::malloc(device, in); rlt::malloc(device, out);
    rlt::init_weights(device, model, rng); rlt::randn(device, in, rng);
    rlt::evaluate(device, model, in, out, buf, rng);
    std::string dp = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/test_dyn_seq_dg.tar";
    rlt::persist::backends::tar::Writer w;
    rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(w)>> wg{"", &w};
    auto mg = rlt::create_group(device, wg, "m"); rlt::save(device, model, mg);
    rlt::persist::backends::tar::finalize(device, w);
    {std::ofstream a(dp, std::ios::binary); a.write(w.buffer.data(), w.buffer.size());}
    auto td = helpers::load_tar(dp);
    rlt::persist::backends::tar::ReaderGroup<rlt::persist::backends::tar::ReaderGroupSpecification<TI>> rg{"", td.data(), static_cast<TI>(td.size())};
    auto dg = rlt::get_group(device, rg, "m");
    rlt::dyn::Layer<TI> dm; rlt::load(device, dm, dg);
    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> di, d_out;
    TI dis[] = {SL, BS, ID}; rlt::dyn::set_shape(di, (TI)3, dis); di.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, di);
    TI dos[] = {SL, BS, (TI)8}; rlt::dyn::set_shape(d_out, (TI)3, dos); d_out.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, d_out);
    auto im = rlt::matrix_view(device, in);
    for(TI i = 0; i < SL*BS; i++) for(TI j = 0; j < ID; j++) rlt::dyn::set(device, di, i*ID+j, rlt::get(im, i, j));
    rlt::dyn::Buffer<TI> db; helpers::setup_buffer(db, dm, di); rlt::malloc(device, db);
    ASSERT_TRUE(rlt::evaluate(device, dm, di, d_out, db));
    auto om = rlt::matrix_view(device, out);
    T md = 0;
    for(TI i = 0; i < SL*BS; i++) for(TI j = 0; j < 8; j++){
        T sv = rlt::get(om, i, j);
        T dv = rlt::dyn::get(device, d_out, i*8+j);
        T d = std::abs(sv - dv);
        if(i < 3 && j < 3) std::cout << "  out[" << i << "][" << j << "] static=" << sv << " dyn=" << dv << " diff=" << d << std::endl;
        if(d > md) md = d;
    }
    std::cout << "Sequential Dense+GRU max diff: " << md << std::endl;
    ASSERT_NEAR(md, 0, 1e-5);
    rlt::free(device, di); rlt::free(device, d_out); rlt::free(device, db); rlt::free(device, dm);
    rlt::free(device, model); rlt::free(device, buf); rlt::free(device, in); rlt::free(device, out);
}

TEST(TEST_DYN, sequential_dense_gru_dense){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);
    using D1C = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 10, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using GC = rlt::nn::layers::gru::Configuration<TYPE_POLICY, TI, 20>;
    using D2C = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 7, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using CHAIN = rlt::nn_models::sequential::Module<rlt::nn::layers::dense::BindConfiguration<D1C>,
        rlt::nn_models::sequential::Module<rlt::nn::layers::gru::BindConfiguration<GC>,
        rlt::nn_models::sequential::Module<rlt::nn::layers::dense::BindConfiguration<D2C>>>>;
    constexpr TI SL2 = 5, BS2 = 3, ID2 = 10, OD2 = 7;
    using M2 = rlt::nn_models::sequential::Build<rlt::nn::capability::Forward<>, CHAIN, rlt::tensor::Shape<TI, SL2, BS2, ID2>>;
    M2 model2; M2::Buffer<> buf2;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename M2::INPUT_SHAPE>> in2;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename M2::OUTPUT_SHAPE>> out2;
    rlt::malloc(device, model2); rlt::malloc(device, buf2); rlt::malloc(device, in2); rlt::malloc(device, out2);
    rlt::init_weights(device, model2, rng); rlt::randn(device, in2, rng);
    rlt::evaluate(device, model2, in2, out2, buf2, rng);
    std::string dp = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/test_dyn_seq_dgd.tar";
    rlt::persist::backends::tar::Writer w;
    rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(w)>> wg{"", &w};
    auto mg = rlt::create_group(device, wg, "m"); rlt::save(device, model2, mg);
    rlt::persist::backends::tar::finalize(device, w);
    {std::ofstream a(dp, std::ios::binary); a.write(w.buffer.data(), w.buffer.size());}
    auto td = helpers::load_tar(dp);
    rlt::persist::backends::tar::ReaderGroup<rlt::persist::backends::tar::ReaderGroupSpecification<TI>> rg{"", td.data(), static_cast<TI>(td.size())};
    auto dg = rlt::get_group(device, rg, "m");
    rlt::dyn::Layer<TI> dm; rlt::load(device, dm, dg);
    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> di, d_out;
    TI dis[] = {SL2, BS2, ID2}; rlt::dyn::set_shape(di, (TI)3, dis); di.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, di);
    TI dos[] = {SL2, BS2, OD2}; rlt::dyn::set_shape(d_out, (TI)3, dos); d_out.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, d_out);
    auto im = rlt::matrix_view(device, in2);
    for(TI i = 0; i < SL2*BS2; i++) for(TI j = 0; j < ID2; j++) rlt::dyn::set(device, di, i*ID2+j, rlt::get(im, i, j));
    rlt::dyn::Buffer<TI> db; helpers::setup_buffer(db, dm, di); rlt::malloc(device, db);
    ASSERT_TRUE(rlt::evaluate(device, dm, di, d_out, db));
    auto om = rlt::matrix_view(device, out2);
    T md = 0;
    for(TI i = 0; i < SL2*BS2; i++) for(TI j = 0; j < OD2; j++){
        T d = std::abs(rlt::get(om, i, j) - rlt::dyn::get(device, d_out, i*OD2+j));
        if(d > md) md = d;
    }
    std::cout << "Sequential Dense+GRU+Dense max diff: " << md << std::endl;
    ASSERT_NEAR(md, 0, 1e-5);
    rlt::free(device, di); rlt::free(device, d_out); rlt::free(device, db); rlt::free(device, dm);
    rlt::free(device, model2); rlt::free(device, buf2); rlt::free(device, in2); rlt::free(device, out2);
}

TEST(TEST_DYN, mlp_standalone){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);
    using MLP_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, 15, 3, 7, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using MLP_CAPABILITY = rlt::nn::capability::Forward<>;
    using MLP_INPUT_SHAPE = rlt::tensor::Shape<TI, 4, 20>;
    using MLP_TYPE = rlt::nn_models::mlp::NeuralNetworkForward<rlt::nn_models::mlp::Specification<MLP_CONFIG, MLP_CAPABILITY, MLP_INPUT_SHAPE>>;
    MLP_TYPE mlp;
    rlt::malloc(device, mlp);
    rlt::init_weights(device, mlp, rng);
    rlt::Tensor<rlt::tensor::Specification<T, TI, MLP_INPUT_SHAPE>> mlp_in;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename MLP_TYPE::OUTPUT_SHAPE>> mlp_out;
    rlt::malloc(device, mlp_in); rlt::malloc(device, mlp_out);
    rlt::randn(device, mlp_in, rng);
    typename MLP_TYPE::Buffer<> mlp_buf;
    rlt::malloc(device, mlp_buf);
    rlt::evaluate(device, mlp, mlp_in, mlp_out, mlp_buf, rng);
    std::string dp = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/test_dyn_mlp.tar";
    rlt::persist::backends::tar::Writer w;
    rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(w)>> wg{"", &w};
    auto mg = rlt::create_group(device, wg, "m"); rlt::save(device, mlp, mg);
    rlt::persist::backends::tar::finalize(device, w);
    {std::ofstream a(dp, std::ios::binary); a.write(w.buffer.data(), w.buffer.size());}
    auto td = helpers::load_tar(dp);
    rlt::persist::backends::tar::ReaderGroup<rlt::persist::backends::tar::ReaderGroupSpecification<TI>> rg{"", td.data(), static_cast<TI>(td.size())};
    auto dg = rlt::get_group(device, rg, "m");
    rlt::dyn::Layer<TI> dm; rlt::load(device, dm, dg);
    ASSERT_EQ(dm.type, rlt::dyn::LayerType::MLP);
    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> di, d_out;
    TI dis[] = {(TI)4, (TI)20}; rlt::dyn::set_shape(di, (TI)2, dis); di.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, di);
    constexpr TI MLP_OUTPUT_DIM = 15;
    TI dos[] = {(TI)4, MLP_OUTPUT_DIM}; rlt::dyn::set_shape(d_out, (TI)2, dos); d_out.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, d_out);
    auto im = rlt::matrix_view(device, mlp_in);
    for(TI i = 0; i < 4; i++) for(TI j = 0; j < 20; j++) rlt::dyn::set(device, di, i*20+j, rlt::get(im, i, j));
    rlt::dyn::Buffer<TI> db; helpers::setup_buffer(db, dm, di); rlt::malloc(device, db);
    ASSERT_TRUE(rlt::evaluate(device, dm, di, d_out, db));
    auto om = rlt::matrix_view(device, mlp_out);
    T md = 0;
    for(TI i = 0; i < 4; i++) for(TI j = 0; j < MLP_OUTPUT_DIM; j++){
        T d = std::abs(rlt::get(om, i, j) - rlt::dyn::get(device, d_out, i*MLP_OUTPUT_DIM+j));
        if(d > md) md = d;
    }
    std::cout << "MLP standalone max diff: " << md << std::endl;
    ASSERT_NEAR(md, 0, 1e-5);
    rlt::free(device, di); rlt::free(device, d_out); rlt::free(device, db); rlt::free(device, dm);
    rlt::free(device, mlp); rlt::free(device, mlp_buf); rlt::free(device, mlp_in); rlt::free(device, mlp_out);
}

TEST(TEST_DYN, sequential_dense_gru_mlp){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);

    using INPUT_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 10, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using INPUT_LAYER = rlt::nn::layers::dense::BindConfiguration<INPUT_LAYER_CONFIG>;
    using GRU_LAYER_CONFIG = rlt::nn::layers::gru::Configuration<TYPE_POLICY, TI, 20>;
    using GRU_LAYER = rlt::nn::layers::gru::BindConfiguration<GRU_LAYER_CONFIG>;
    using MLP_LAYER_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, 15, 3, 7, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using MLP_LAYER = rlt::nn_models::mlp::BindConfiguration<MLP_LAYER_CONFIG>;
    using CAPABILITY = rlt::nn::capability::Forward<>;
    using MODULE_CHAIN = rlt::nn_models::sequential::Module<INPUT_LAYER, rlt::nn_models::sequential::Module<GRU_LAYER, rlt::nn_models::sequential::Module<MLP_LAYER>>>;

    constexpr TI SEQ_LEN = 5;
    constexpr TI BATCH_SIZE = 3;
    constexpr TI INPUT_DIM = 10;
    using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQ_LEN, BATCH_SIZE, INPUT_DIM>;
    using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename MODEL::INPUT_SHAPE>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename MODEL::OUTPUT_SHAPE>> output_static;
    MODEL model;
    MODEL::Buffer<> buffer;
    rlt::malloc(device, input);
    rlt::malloc(device, output_static);
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);
    rlt::init_weights(device, model, rng);
    rlt::randn(device, input, rng);
    rlt::evaluate(device, model, input, output_static, buffer, rng);

    // Save to TAR
    std::string data_path = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/test_dyn_sequential.tar";
    rlt::persist::backends::tar::Writer writer;
    rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(writer)>> writer_group{"", &writer};
    auto model_group = rlt::create_group(device, writer_group, "model");
    rlt::save(device, model, model_group);
    rlt::persist::backends::tar::finalize(device, writer);
    std::ofstream archive(data_path, std::ios::binary);
    archive.write(writer.buffer.data(), writer.buffer.size());
    archive.close();

    // Load into dyn
    auto tar_data = helpers::load_tar(data_path);
    rlt::persist::backends::tar::ReaderGroup<rlt::persist::backends::tar::ReaderGroupSpecification<TI>> reader_group{"", tar_data.data(), static_cast<TI>(tar_data.size())};
    auto dyn_model_group = rlt::get_group(device, reader_group, "model");
    rlt::dyn::Layer<TI> dyn_model;
    rlt::load(device, dyn_model, dyn_model_group);
    ASSERT_EQ(dyn_model.type, rlt::dyn::LayerType::SEQUENTIAL);

    // Prepare dyn input
    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> dyn_input, dyn_output;
    TI input_shape[] = {SEQ_LEN, BATCH_SIZE, INPUT_DIM};
    rlt::dyn::set_shape(dyn_input, (TI)3, input_shape);
    dyn_input.type = rlt::dyn::Type::FLOAT32;
    rlt::malloc(device, dyn_input);
    // Copy input data (flat copy)
    auto input_mat = rlt::matrix_view(device, input);
    for(TI i = 0; i < SEQ_LEN * BATCH_SIZE; i++){
        for(TI j = 0; j < INPUT_DIM; j++){
            rlt::dyn::set(device, dyn_input, i * INPUT_DIM + j, rlt::get(input_mat, i, j));
        }
    }

    // Allocate output (MLP config: OUTPUT_DIM=15, HIDDEN_DIM=7)
    constexpr TI OUTPUT_DIM = 15;
    TI output_shape[] = {SEQ_LEN, BATCH_SIZE, OUTPUT_DIM};
    rlt::dyn::set_shape(dyn_output, (TI)3, output_shape);
    dyn_output.type = rlt::dyn::Type::FLOAT32;
    rlt::malloc(device, dyn_output);

    rlt::dyn::Buffer<TI> dyn_buffer;
    helpers::setup_buffer(dyn_buffer, dyn_model, dyn_input);
    rlt::malloc(device, dyn_buffer);

    ASSERT_TRUE(rlt::evaluate(device, dyn_model, dyn_input, dyn_output, dyn_buffer));

    // Compare outputs
    auto output_mat = rlt::matrix_view(device, output_static);
    T max_diff = 0;
    for(TI i = 0; i < SEQ_LEN * BATCH_SIZE; i++){
        for(TI j = 0; j < OUTPUT_DIM; j++){
            T static_val = rlt::get(output_mat, i, j);
            T dyn_val = rlt::dyn::get(device, dyn_output, i * OUTPUT_DIM + j);
            T diff = std::abs(static_val - dyn_val);
            if(diff > max_diff) max_diff = diff;
        }
    }
    std::cout << "Sequential max diff: " << max_diff << std::endl;
    ASSERT_NEAR(max_diff, 0, 1e-5);

    rlt::free(device, dyn_input);
    rlt::free(device, dyn_output);
    rlt::free(device, dyn_buffer);
    rlt::free(device, dyn_model);
    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, input);
    rlt::free(device, output_static);
}

TEST(TEST_DYN, gru_step_only){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 55);
    constexpr TI GI2 = 4, GH2 = 5, GB2 = 2, NS = 3;
    using GC2 = rlt::nn::layers::gru::Configuration<TYPE_POLICY, TI, GH2>;
    using GS2 = rlt::nn::layers::gru::Specification<GC2, rlt::nn::capability::Forward<>, rlt::tensor::Shape<TI, NS, GB2, GI2>>;
    rlt::nn::layers::gru::LayerForward<GS2> gru;
    rlt::malloc(device, gru);
    rlt::init_weights(device, gru, rng);
    typename rlt::nn::layers::gru::LayerForward<GS2>::template State<true> static_state;
    typename rlt::nn::layers::gru::LayerForward<GS2>::template Buffer<> static_buf;
    rlt::malloc(device, static_state); rlt::malloc(device, static_buf);
    rlt::reset(device, gru, static_state, rng);
    // Save to TAR
    std::string dp = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/test_dyn_gru_step.tar";
    rlt::persist::backends::tar::Writer w;
    rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(w)>> wg{"", &w};
    auto lg = rlt::create_group(device, wg, "layer");
    rlt::save(device, gru, lg);
    rlt::persist::backends::tar::finalize(device, w);
    {std::ofstream a(dp, std::ios::binary); a.write(w.buffer.data(), w.buffer.size());}
    auto td = helpers::load_tar(dp);
    rlt::persist::backends::tar::ReaderGroup<rlt::persist::backends::tar::ReaderGroupSpecification<TI>> rg{"", td.data(), static_cast<TI>(td.size())};
    auto dlg = rlt::get_group(device, rg, "layer");
    rlt::dyn::Layer<TI> dl; rlt::load(device, dl, dlg);
    rlt::dyn::State<TI> ds; ds.batch_size = GB2; ds.layer = &dl; rlt::malloc(device, ds); rlt::reset(device, dl, ds);
    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> di, d_out;
    TI dis[] = {GB2, GI2}; rlt::dyn::set_shape(di, (TI)2, dis); di.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, di);
    TI dos[] = {GB2, GH2}; rlt::dyn::set_shape(d_out, (TI)2, dos); d_out.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, d_out);
    rlt::dyn::Buffer<TI> db; helpers::setup_buffer(db, dl, di); rlt::malloc(device, db);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, GB2, GI2>>> si;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, GB2, GH2>>> s_out;
    rlt::malloc(device, si); rlt::malloc(device, s_out);
    T omd = 0;
    for(TI step = 0; step < NS; step++){
        rlt::randn(device, si, rng);
        auto sim = rlt::matrix_view(device, si);
        for(TI b = 0; b < GB2; b++) for(TI i = 0; i < GI2; i++) rlt::dyn::set(device, di, b*GI2+i, rlt::get(sim, b, i));
        rlt::evaluate_step(device, gru, si, static_state, s_out, static_buf, rng);
        ASSERT_TRUE(rlt::evaluate_step(device, dl, di, ds, d_out, db));
        auto som = rlt::matrix_view(device, s_out);
        T smd = 0;
        for(TI b = 0; b < GB2; b++) for(TI h = 0; h < GH2; h++){
            T d = std::abs(rlt::get(som, b, h) - rlt::dyn::get(device, d_out, b*GH2+h));
            if(d > smd) smd = d;
        }
        if(smd > omd) omd = smd;
    }
    std::cout << "GRU step-only max diff over " << NS << " steps: " << omd << std::endl;
    ASSERT_NEAR(omd, 0, 1e-5);
    rlt::free(device, di); rlt::free(device, d_out); rlt::free(device, db); rlt::free(device, ds); rlt::free(device, dl);
    rlt::free(device, gru); rlt::free(device, static_state); rlt::free(device, static_buf); rlt::free(device, si); rlt::free(device, s_out);
}

TEST(TEST_DYN, evaluate_step_gru){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 123);

    using INPUT_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 8, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using INPUT_LAYER = rlt::nn::layers::dense::BindConfiguration<INPUT_LAYER_CONFIG>;
    using GRU_LAYER_CONFIG = rlt::nn::layers::gru::Configuration<TYPE_POLICY, TI, 12>;
    using GRU_LAYER = rlt::nn::layers::gru::BindConfiguration<GRU_LAYER_CONFIG>;
    using OUTPUT_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 5, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using OUTPUT_LAYER = rlt::nn::layers::dense::BindConfiguration<OUTPUT_LAYER_CONFIG>;
    using CAPABILITY = rlt::nn::capability::Forward<>;
    using MODULE_CHAIN = rlt::nn_models::sequential::Module<INPUT_LAYER, rlt::nn_models::sequential::Module<GRU_LAYER, rlt::nn_models::sequential::Module<OUTPUT_LAYER>>>;

    constexpr TI BATCH_SIZE = 2;
    constexpr TI INPUT_DIM = 6;
    constexpr TI OUTPUT_DIM = 5;
    constexpr TI NUM_STEPS = 10;
    using FULL_INPUT_SHAPE = rlt::tensor::Shape<TI, NUM_STEPS, BATCH_SIZE, INPUT_DIM>;
    using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, FULL_INPUT_SHAPE>;
    using STEP_INPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, INPUT_DIM>;

    MODEL model;
    MODEL::Buffer<> buffer;
    typename MODEL::template State<true> state;
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);
    rlt::malloc(device, state);
    rlt::init_weights(device, model, rng);
    rlt::reset(device, model, state, rng);

    // Save to TAR
    std::string data_path = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/test_dyn_eval_step.tar";
    rlt::persist::backends::tar::Writer writer;
    rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(writer)>> writer_group{"", &writer};
    auto model_group = rlt::create_group(device, writer_group, "model");
    rlt::save(device, model, model_group);
    rlt::persist::backends::tar::finalize(device, writer);
    std::ofstream archive(data_path, std::ios::binary);
    archive.write(writer.buffer.data(), writer.buffer.size());
    archive.close();

    // Load into dyn
    auto tar_data = helpers::load_tar(data_path);
    rlt::persist::backends::tar::ReaderGroup<rlt::persist::backends::tar::ReaderGroupSpecification<TI>> reader_group{"", tar_data.data(), static_cast<TI>(tar_data.size())};
    auto dyn_model_group = rlt::get_group(device, reader_group, "model");
    rlt::dyn::Layer<TI> dyn_model;
    rlt::load(device, dyn_model, dyn_model_group);

    // Prepare dyn state and buffer
    rlt::dyn::State<TI> dyn_state;
    dyn_state.batch_size = BATCH_SIZE;
    dyn_state.layer = &dyn_model;
    rlt::malloc(device, dyn_state);
    rlt::reset(device, dyn_model, dyn_state);

    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> dyn_step_input, dyn_step_output;
    TI step_input_shape[] = {BATCH_SIZE, INPUT_DIM};
    rlt::dyn::set_shape(dyn_step_input, (TI)2, step_input_shape);
    dyn_step_input.type = rlt::dyn::Type::FLOAT32;
    rlt::malloc(device, dyn_step_input);
    TI step_output_shape[] = {BATCH_SIZE, OUTPUT_DIM};
    rlt::dyn::set_shape(dyn_step_output, (TI)2, step_output_shape);
    dyn_step_output.type = rlt::dyn::Type::FLOAT32;
    rlt::malloc(device, dyn_step_output);

    rlt::dyn::Buffer<TI> dyn_buffer;
    helpers::setup_buffer(dyn_buffer, dyn_model, dyn_step_input);
    rlt::malloc(device, dyn_buffer);

    rlt::Tensor<rlt::tensor::Specification<T, TI, STEP_INPUT_SHAPE>> step_input;
    using STEP_OUTPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, OUTPUT_DIM>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, STEP_OUTPUT_SHAPE>> step_output_static;
    rlt::malloc(device, step_input);
    rlt::malloc(device, step_output_static);

    T overall_max_diff = 0;
    for(TI step = 0; step < NUM_STEPS; step++){
        rlt::randn(device, step_input, rng);
        auto input_mat = rlt::matrix_view(device, step_input);
        for(TI b = 0; b < BATCH_SIZE; b++){
            for(TI i = 0; i < INPUT_DIM; i++){
                rlt::dyn::set(device, dyn_step_input, b * INPUT_DIM + i, rlt::get(input_mat, b, i));
            }
        }

        rlt::evaluate_step(device, model, step_input, state, step_output_static, buffer, rng);
        ASSERT_TRUE(rlt::evaluate_step(device, dyn_model, dyn_step_input, dyn_state, dyn_step_output, dyn_buffer));

        auto output_mat = rlt::matrix_view(device, step_output_static);
        T step_max_diff = 0;
        for(TI b = 0; b < BATCH_SIZE; b++){
            for(TI j = 0; j < OUTPUT_DIM; j++){
                T static_val = rlt::get(output_mat, b, j);
                T dyn_val = rlt::dyn::get(device, dyn_step_output, b * OUTPUT_DIM + j);
                T diff = std::abs(static_val - dyn_val);
                if(diff > step_max_diff) step_max_diff = diff;
            }
        }
        if(step_max_diff > overall_max_diff) overall_max_diff = step_max_diff;
    }
    std::cout << "evaluate_step max diff over " << NUM_STEPS << " steps: " << overall_max_diff << std::endl;
    ASSERT_NEAR(overall_max_diff, 0, 1e-5);

    rlt::free(device, dyn_step_input);
    rlt::free(device, dyn_step_output);
    rlt::free(device, dyn_buffer);
    rlt::free(device, dyn_state);
    rlt::free(device, dyn_model);
    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, state);
    rlt::free(device, step_input);
    rlt::free(device, step_output_static);
}

TEST(TEST_DYN, basic_cnn){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);
    // Conv2d(3→8, 3x3, stride=1, pad=1, RELU) → MaxPool2d(2x2, stride=2) → Flatten → Dense(→5)
    using CONV_CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 8, 3, 3, 1, 1, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using POOL_CONFIG = rlt::nn::layers::max_pool2d::Configuration<TYPE_POLICY, TI, 2, 2, 2, 2>;
    using FLAT_CONFIG = rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>;
    using FC_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 5, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using CHAIN = rlt::nn_models::sequential::Module<
        rlt::nn::layers::conv2d::BindConfiguration<CONV_CONFIG>,
        rlt::nn_models::sequential::Module<rlt::nn::layers::max_pool2d::BindConfiguration<POOL_CONFIG>,
        rlt::nn_models::sequential::Module<rlt::nn::layers::flatten::BindConfiguration<FLAT_CONFIG>,
        rlt::nn_models::sequential::Module<rlt::nn::layers::dense::BindConfiguration<FC_CONFIG>>>>>;
    // Input: (1, 8, 8, 3)
    using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, 8, 8, 3>;
    using MODEL = rlt::nn_models::sequential::Build<rlt::nn::capability::Forward<>, CHAIN, INPUT_SHAPE>;
    MODEL model; MODEL::Buffer<> buf;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> in;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename MODEL::OUTPUT_SHAPE>> out;
    rlt::malloc(device, model); rlt::malloc(device, buf); rlt::malloc(device, in); rlt::malloc(device, out);
    rlt::init_weights(device, model, rng); rlt::randn(device, in, rng);
    rlt::evaluate(device, model, in, out, buf, rng);
    // Save to TAR
    std::string dp = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/test_dyn_basic_cnn.tar";
    rlt::persist::backends::tar::Writer w;
    rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(w)>> wg{"", &w};
    auto mg = rlt::create_group(device, wg, "m"); rlt::save(device, model, mg);
    rlt::persist::backends::tar::finalize(device, w);
    {std::ofstream a(dp, std::ios::binary); a.write(w.buffer.data(), w.buffer.size());}
    // Load into dyn
    auto td = helpers::load_tar(dp);
    rlt::persist::backends::tar::ReaderGroup<rlt::persist::backends::tar::ReaderGroupSpecification<TI>> rg{"", td.data(), static_cast<TI>(td.size())};
    auto dg = rlt::get_group(device, rg, "m");
    rlt::dyn::Layer<TI> dm; rlt::load(device, dm, dg);
    ASSERT_EQ(dm.type, rlt::dyn::LayerType::SEQUENTIAL);
    // Prepare dyn input
    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> di, d_out;
    TI dis[] = {(TI)1, (TI)8, (TI)8, (TI)3}; rlt::dyn::set_shape(di, (TI)4, dis); di.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, di);
    auto im = rlt::matrix_view(device, in);
    for(TI i = 0; i < 1 * 8 * 8 * 3; i++) rlt::dyn::set(device, di, i, rlt::get_flat(device, in, i));
    // Output: (1, 5) from flatten→dense
    constexpr TI CNN_OUTPUT_DIM = 5;
    TI dos[] = {(TI)1, CNN_OUTPUT_DIM}; rlt::dyn::set_shape(d_out, (TI)2, dos); d_out.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, d_out);
    rlt::dyn::Buffer<TI> db; helpers::setup_buffer(db, dm, di); rlt::malloc(device, db);
    ASSERT_TRUE(rlt::evaluate(device, dm, di, d_out, db));
    auto om = rlt::matrix_view(device, out);
    T md = 0;
    for(TI j = 0; j < CNN_OUTPUT_DIM; j++){
        T d = std::abs(rlt::get(om, 0, j) - rlt::dyn::get(device, d_out, j));
        if(d > md) md = d;
    }
    std::cout << "Basic CNN max diff: " << md << std::endl;
    ASSERT_NEAR(md, 0, 1e-4);
    rlt::free(device, di); rlt::free(device, d_out); rlt::free(device, db); rlt::free(device, dm);
    rlt::free(device, model); rlt::free(device, buf); rlt::free(device, in); rlt::free(device, out);
}

TEST(TEST_DYN, resnet_block){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);
    // Single resnet block: 64→64, stride=1 (no downsample)
    using RB_CONFIG = rlt::nn::layers::resnet_block::Configuration<TYPE_POLICY, TI, 64, 1>;
    using RB_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, 8, 8, 64>;
    using RB_SPEC = rlt::nn::layers::resnet_block::Specification<RB_CONFIG, rlt::nn::capability::Forward<>, RB_INPUT_SHAPE>;
    rlt::nn::layers::resnet_block::LayerForward<RB_SPEC> rb;
    rlt::malloc(device, rb);
    rlt::init_weights(device, rb, rng);
    rlt::Tensor<rlt::tensor::Specification<T, TI, RB_INPUT_SHAPE>> rb_in;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename RB_SPEC::OUTPUT_SHAPE>> rb_out;
    typename rlt::nn::layers::resnet_block::LayerForward<RB_SPEC>::template Buffer<> rb_buf;
    rlt::malloc(device, rb_in); rlt::malloc(device, rb_out);
    rlt::malloc(device, rb_buf);
    rlt::randn(device, rb_in, rng);
    rlt::evaluate(device, rb, rb_in, rb_out, rb_buf, rng);
    std::string dp = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/test_dyn_resnet_block.tar";
    rlt::persist::backends::tar::Writer w;
    rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(w)>> wg{"", &w};
    auto mg = rlt::create_group(device, wg, "m"); rlt::save(device, rb, mg);
    rlt::persist::backends::tar::finalize(device, w);
    {std::ofstream a(dp, std::ios::binary); a.write(w.buffer.data(), w.buffer.size());}
    auto td = helpers::load_tar(dp);
    rlt::persist::backends::tar::ReaderGroup<rlt::persist::backends::tar::ReaderGroupSpecification<TI>> rg{"", td.data(), static_cast<TI>(td.size())};
    auto dg = rlt::get_group(device, rg, "m");
    rlt::dyn::Layer<TI> dm; rlt::load(device, dm, dg);
    std::cout << "Dyn load done" << std::endl;
    ASSERT_EQ(dm.type, rlt::dyn::LayerType::RESNET_BLOCK);
    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> di, d_out;
    TI dis[] = {(TI)1, (TI)8, (TI)8, (TI)64}; rlt::dyn::set_shape(di, (TI)4, dis); di.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, di);
    for(TI i = 0; i < 1*8*8*64; i++) rlt::dyn::set(device, di, i, rlt::get_flat(device, rb_in, i));
    TI dos[] = {(TI)1, (TI)8, (TI)8, (TI)64}; rlt::dyn::set_shape(d_out, (TI)4, dos); d_out.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, d_out);
    rlt::dyn::Buffer<TI> db; helpers::setup_buffer(db, dm, di); rlt::malloc(device, db);
    ASSERT_TRUE(rlt::evaluate(device, dm, di, d_out, db));
    T md = 0;
    for(TI i = 0; i < 1*8*8*64; i++){
        T d = std::abs(rlt::get_flat(device, rb_out, i) - rlt::dyn::get(device, d_out, i));
        if(d > md) md = d;
    }
    std::cout << "ResNet block (no downsample) max diff: " << md << std::endl;
    ASSERT_NEAR(md, 0, 1e-4);
    rlt::free(device, di); rlt::free(device, d_out); rlt::free(device, db); rlt::free(device, dm);
    rlt::free(device, rb); rlt::free(device, rb_buf); rlt::free(device, rb_in); rlt::free(device, rb_out);
}

TEST(TEST_DYN, resnet18){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);
    using CAPABILITY = rlt::nn::capability::Forward<>;
    using MODEL = rlt::nn_models::resnet18::MODEL<TYPE_POLICY, TI, CAPABILITY>;
    MODEL model; MODEL::Buffer<> buf;
    using INPUT_SHAPE = rlt::nn_models::resnet18::INPUT_SHAPE<TYPE_POLICY, TI>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> in;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename MODEL::OUTPUT_SHAPE>> out;
    rlt::malloc(device, model); rlt::malloc(device, buf); rlt::malloc(device, in); rlt::malloc(device, out);
    rlt::init_weights(device, model, rng); rlt::randn(device, in, rng);
    rlt::evaluate(device, model, in, out, buf, rng);
    // Save to TAR
    std::string dp = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/test_dyn_resnet18.tar";
    rlt::persist::backends::tar::Writer w;
    rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(w)>> wg{"", &w};
    auto mg = rlt::create_group(device, wg, "m"); rlt::save(device, model, mg);
    rlt::persist::backends::tar::finalize(device, w);
    {std::ofstream a(dp, std::ios::binary); a.write(w.buffer.data(), w.buffer.size());}
    // Load into dyn
    auto td = helpers::load_tar(dp);
    rlt::persist::backends::tar::ReaderGroup<rlt::persist::backends::tar::ReaderGroupSpecification<TI>> rg{"", td.data(), static_cast<TI>(td.size())};
    auto dg = rlt::get_group(device, rg, "m");
    rlt::dyn::Layer<TI> dm; rlt::load(device, dm, dg);
    ASSERT_EQ(dm.type, rlt::dyn::LayerType::SEQUENTIAL);
    // Prepare dyn input (1, 224, 224, 3)
    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> di, d_out;
    TI dis[] = {(TI)1, (TI)224, (TI)224, (TI)3}; rlt::dyn::set_shape(di, (TI)4, dis); di.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, di);
    for(TI i = 0; i < 1 * 224 * 224 * 3; i++) rlt::dyn::set(device, di, i, rlt::get_flat(device, in, i));
    // Output: (1, 1000)
    constexpr TI RESNET_OUTPUT_DIM = 1000;
    TI dos[] = {(TI)1, RESNET_OUTPUT_DIM}; rlt::dyn::set_shape(d_out, (TI)2, dos); d_out.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, d_out);
    rlt::dyn::Buffer<TI> db; helpers::setup_buffer(db, dm, di); rlt::malloc(device, db);
    ASSERT_TRUE(rlt::evaluate(device, dm, di, d_out, db));
    auto om = rlt::matrix_view(device, out);
    T md = 0;
    for(TI j = 0; j < RESNET_OUTPUT_DIM; j++){
        T d = std::abs(rlt::get(om, 0, j) - rlt::dyn::get(device, d_out, j));
        if(d > md) md = d;
    }
    std::cout << "ResNet18 max diff: " << md << std::endl;
    ASSERT_NEAR(md, 0, 1e-3);
    rlt::free(device, di); rlt::free(device, d_out); rlt::free(device, db); rlt::free(device, dm);
    rlt::free(device, model); rlt::free(device, buf); rlt::free(device, in); rlt::free(device, out);
}

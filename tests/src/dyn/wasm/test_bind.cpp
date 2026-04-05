#include <rl_tools/operations/cpu.h>
#include <rl_tools/persist/backends/hdf5/hdf5.h>
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#include <rl_tools/dyn/persist.h>

#include <emscripten/bind.h>
#include <vector>
#include <cstdio>

namespace rlt = rl_tools;
using DEVICE = rlt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;

static DEVICE device;
static rlt::dyn::Layer<TI> model;
static rlt::dyn::Buffer<TI> buffer;
static TI input_rank = 0;
static TI input_shape_stored[8] = {};
static TI output_dim = 0;
static std::vector<float> output_cache;
static std::vector<rlt::dyn::State<TI>> states;
static bool ready = false;

bool load_model(const std::string& path, emscripten::val js_input_shape){
    rlt::persist::backends::hdf5::File file(path.c_str(), rlt::persist::backends::hdf5::Mode::READ);
    if(file.id < 0) return false;
    bool has_actor = H5Lexists(file.id, "actor", H5P_DEFAULT) > 0;
    auto model_group = rlt::get_group(device, file, has_actor ? "actor" : "model");
    if(!rlt::load(device, model, model_group)) return false;

    unsigned int rank = js_input_shape["length"].as<unsigned int>();
    TI input_shape[8];
    TI total = 1;
    for(unsigned int d = 0; d < rank; d++){
        input_shape[d] = js_input_shape[d].as<TI>();
        input_shape_stored[d] = input_shape[d];
        total *= input_shape[d];
    }
    input_rank = (TI)rank;

    rlt::dyn::propagate_shapes(model, input_shape, (TI)rank, total);
    output_dim = model.output_size;
    output_cache.resize(output_dim);
    buffer.layer = &model;
    rlt::malloc(device, buffer);
    ready = true;
    return true;
}

emscripten::val evaluate_via_val(emscripten::val js_input){
    if(!ready) return emscripten::val::null();
    unsigned int len = js_input["length"].as<unsigned int>();
    std::vector<float> input_data(len);
    for(unsigned int i = 0; i < len; i++){
        input_data[i] = js_input[i].as<float>();
    }
    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> input_tensor, output_tensor;
    rlt::dyn::set_shape(input_tensor, input_rank, input_shape_stored);
    input_tensor.type = rlt::dyn::Type::FLOAT32;
    input_tensor.data = input_data.data();
    TI out_shape[] = {output_dim};
    rlt::dyn::set_shape(output_tensor, (TI)1, out_shape);
    output_tensor.type = rlt::dyn::Type::FLOAT32;
    output_tensor.data = output_cache.data();
    rlt::evaluate(device, model, input_tensor, output_tensor, buffer);
    return emscripten::val(emscripten::typed_memory_view(output_dim, output_cache.data()));
}

emscripten::val evaluate_via_heap(emscripten::val js_input){
    if(!ready) return emscripten::val::null();
    unsigned int len = js_input["length"].as<unsigned int>();
    std::vector<float> input_data(len);
    emscripten::val heap_view = emscripten::val(emscripten::typed_memory_view(len, input_data.data()));
    heap_view.call<void>("set", js_input);
    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> input_tensor, output_tensor;
    rlt::dyn::set_shape(input_tensor, input_rank, input_shape_stored);
    input_tensor.type = rlt::dyn::Type::FLOAT32;
    input_tensor.data = input_data.data();
    TI out_shape[] = {output_dim};
    rlt::dyn::set_shape(output_tensor, (TI)1, out_shape);
    output_tensor.type = rlt::dyn::Type::FLOAT32;
    output_tensor.data = output_cache.data();
    rlt::evaluate(device, model, input_tensor, output_tensor, buffer);
    return emscripten::val(emscripten::typed_memory_view(output_dim, output_cache.data()));
}

int create_state(){
    rlt::dyn::State<TI> state;
    state.batch_size = 1;
    state.layer = &model;
    rlt::malloc(device, state);
    states.push_back(std::move(state));
    return (int)(states.size() - 1);
}

void reset_state(int id){
    if(id >= 0 && id < (int)states.size())
        rlt::reset(device, model, states[id]);
}

emscripten::val evaluate_step_stateful(int state_id, emscripten::val js_input){
    if(!ready || state_id < 0 || state_id >= (int)states.size()) return emscripten::val::null();
    unsigned int len = js_input["length"].as<unsigned int>();
    std::vector<float> input_data(len);
    emscripten::val heap_view = emscripten::val(emscripten::typed_memory_view(len, input_data.data()));
    heap_view.call<void>("set", js_input);
    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> input_tensor, output_tensor;
    rlt::dyn::set_shape(input_tensor, input_rank, input_shape_stored);
    input_tensor.type = rlt::dyn::Type::FLOAT32;
    input_tensor.data = input_data.data();
    TI out_shape[] = {output_dim};
    rlt::dyn::set_shape(output_tensor, (TI)1, out_shape);
    output_tensor.type = rlt::dyn::Type::FLOAT32;
    output_tensor.data = output_cache.data();
    rlt::evaluate_step(device, model, input_tensor, states[state_id], output_tensor, buffer);
    return emscripten::val(emscripten::typed_memory_view(output_dim, output_cache.data()));
}

EMSCRIPTEN_BINDINGS(test_bind){
    emscripten::function("load_model", &load_model);
    emscripten::function("evaluate_via_val", &evaluate_via_val);
    emscripten::function("evaluate_via_heap", &evaluate_via_heap);
    emscripten::function("create_state", &create_state);
    emscripten::function("reset_state", &reset_state);
    emscripten::function("evaluate_step_stateful", &evaluate_step_stateful);
}

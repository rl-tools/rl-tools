#pragma once

#include <rl_tools/operations/cpu.h>
#include <rl_tools/persist/backends/hdf5/hdf5.h>
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#include <rl_tools/dyn/persist.h>

#include <emscripten/bind.h>

#include <vector>
#include <string>
#include <cmath>
#include <cstdio>

namespace rlt = rl_tools;

struct DynInference {
    using DEVICE = rlt::devices::DefaultCPU;
    using TI = typename DEVICE::index_t;

    DEVICE device;
    rlt::dyn::Layer<TI> model;
    rlt::dyn::Buffer<TI> buffer;
    std::vector<rlt::dyn::State<TI>> states;
    TI input_dim = 0;
    TI output_dim = 0;
    std::string checkpoint_name;
    std::string meta_json;
    bool loaded = false;
    std::vector<float> output_cache;
    std::vector<std::vector<float>> example_inputs_data;
    std::vector<std::vector<TI>> example_inputs_shape;
    std::vector<float> example_output_data;
    std::vector<TI> example_output_shape;

    void propagate_runtime_shapes() {
        if(model.type == rlt::dyn::LayerType::PARALLEL && !example_inputs_data.empty()){
            rlt::dyn::TensorTuple<TI> tuple;
            TI n = (TI)example_inputs_data.size();
            if(n > rlt::dyn::TensorTuple<TI>::MAX_TENSORS) n = rlt::dyn::TensorTuple<TI>::MAX_TENSORS;
            tuple.num_tensors = n;
            for(TI i = 0; i < n; i++){
                rlt::dyn::set_shape(tuple.tensors[i],
                    (TI)example_inputs_shape[i].size(),
                    example_inputs_shape[i].data());
                tuple.tensors[i].type = rlt::dyn::Type::FLOAT32;
                tuple.tensors[i].data = example_inputs_data[i].data();
                tuple.tensors[i].capacity = example_inputs_data[i].size();
            }
            rlt::dyn::propagate_shapes(model, tuple);
        }
        else{
            TI legacy_shape[] = {(TI)1, input_dim};
            rlt::dyn::propagate_shapes(model, legacy_shape, (TI)2);
        }
        output_dim = model.output_shape[model.output_rank - 1];
    }

    void propagate_example_shapes() {
        if(model.type == rlt::dyn::LayerType::PARALLEL && !example_inputs_data.empty()){
            rlt::dyn::TensorTuple<TI> tuple;
            TI n = (TI)example_inputs_data.size();
            if(n > rlt::dyn::TensorTuple<TI>::MAX_TENSORS) n = rlt::dyn::TensorTuple<TI>::MAX_TENSORS;
            tuple.num_tensors = n;
            for(TI i = 0; i < n; i++){
                rlt::dyn::set_shape(tuple.tensors[i],
                    (TI)example_inputs_shape[i].size(),
                    example_inputs_shape[i].data());
                tuple.tensors[i].type = rlt::dyn::Type::FLOAT32;
                tuple.tensors[i].data = example_inputs_data[i].data();
                tuple.tensors[i].capacity = example_inputs_data[i].size();
            }
            rlt::dyn::propagate_shapes(model, tuple);
        }
        else if(!example_inputs_shape.empty()){
            rlt::dyn::propagate_shapes(model,
                example_inputs_shape[0].data(),
                (TI)example_inputs_shape[0].size());
        }
    }

    static bool read_dataset_float(hid_t parent, const char* name, std::vector<TI>& shape_out, std::vector<float>& data_out) {
        if(H5Lexists(parent, name, H5P_DEFAULT) <= 0) return false;
        hid_t ds = H5Dopen2(parent, name, H5P_DEFAULT);
        if(ds < 0) return false;
        hid_t space = H5Dget_space(ds);
        int rank = H5Sget_simple_extent_ndims(space);
        hsize_t dims[8];
        if(rank < 0 || rank > 8){ H5Sclose(space); H5Dclose(ds); return false; }
        H5Sget_simple_extent_dims(space, dims, nullptr);
        shape_out.resize(rank);
        TI total = 1;
        for(int d = 0; d < rank; d++){ shape_out[d] = (TI)dims[d]; total *= (TI)dims[d]; }
        data_out.resize(total);
        H5Dread(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_out.data());
        H5Sclose(space);
        H5Dclose(ds);
        return true;
    }

    bool load(const std::string& path) {
        if(loaded) destroy();
        rlt::persist::backends::hdf5::File file(path.c_str(), rlt::persist::backends::hdf5::Mode::READ);
        if(file.id < 0) return false;

        auto actor_group = rlt::get_group(device, file, "actor");

        char attr_buf[4096];
        rlt::persist::backends::hdf5::detail::read_string_attribute(actor_group.id, "checkpoint_name", attr_buf, sizeof(attr_buf));
        checkpoint_name = attr_buf;
        rlt::persist::backends::hdf5::detail::read_string_attribute(actor_group.id, "meta", attr_buf, sizeof(attr_buf));
        meta_json = attr_buf;

        if(!rlt::load(device, model, actor_group)) return false;

        if(H5Lexists(file.id, "example", H5P_DEFAULT) > 0){
            hid_t example_group_id = H5Gopen2(file.id, "example", H5P_DEFAULT);
            if(example_group_id >= 0){
                if(H5Lexists(example_group_id, "inputs", H5P_DEFAULT) > 0){
                    hid_t inputs_group_id = H5Gopen2(example_group_id, "inputs", H5P_DEFAULT);
                    if(inputs_group_id >= 0){
                        for(TI i = 0; i < rlt::dyn::TensorTuple<TI>::MAX_TENSORS; i++){
                            char name[16]; std::snprintf(name, sizeof(name), "%lu", (unsigned long)i);
                            std::vector<TI> shape; std::vector<float> data;
                            if(!read_dataset_float(inputs_group_id, name, shape, data)) break;
                            example_inputs_shape.push_back(std::move(shape));
                            example_inputs_data.push_back(std::move(data));
                        }
                        H5Gclose(inputs_group_id);
                    }
                }
                else{
                    std::vector<TI> shape; std::vector<float> data;
                    if(read_dataset_float(example_group_id, "input", shape, data)){
                        example_inputs_shape.push_back(std::move(shape));
                        example_inputs_data.push_back(std::move(data));
                    }
                }

                bool output_loaded = false;
                if(H5Lexists(example_group_id, "outputs", H5P_DEFAULT) > 0){
                    hid_t outputs_group_id = H5Gopen2(example_group_id, "outputs", H5P_DEFAULT);
                    if(outputs_group_id >= 0){
                        output_loaded = read_dataset_float(outputs_group_id, "0", example_output_shape, example_output_data);
                        H5Gclose(outputs_group_id);
                    }
                }
                if(!output_loaded){
                    read_dataset_float(example_group_id, "output", example_output_shape, example_output_data);
                }
                H5Gclose(example_group_id);
            }
        }

        if(!example_inputs_shape.empty()){
            const auto& sh = example_inputs_shape[0];
            TI feature_start = (sh.size() >= 3) ? 2 : (sh.size() >= 2 ? 1 : 0);
            TI flat = 1;
            for(size_t d = feature_start; d < sh.size(); d++) flat *= sh[d];
            input_dim = flat;
        }

        propagate_runtime_shapes();

        buffer.layer = &model;
        rlt::malloc(device, buffer);

        output_cache.resize(model.output_size);
        loaded = true;
        return true;
    }

    std::string get_checkpoint_name() const { return checkpoint_name; }
    std::string get_meta() const { return meta_json; }
    int get_input_dim() const { return (int)input_dim; }
    int get_output_dim() const { return (int)output_dim; }

    int get_num_branches() const {
        if(model.type == rlt::dyn::LayerType::PARALLEL && model.data){
            auto& p = model.template as<const rlt::dyn::layers::Parallel<TI>>();
            if(p.num_branches > 0) return (int)p.num_branches;
        }
        return 1;
    }

    emscripten::val get_input_dims() const {
        emscripten::val arr = emscripten::val::array();
        if(model.type == rlt::dyn::LayerType::PARALLEL && model.data){
            auto& p = model.template as<const rlt::dyn::layers::Parallel<TI>>();
            for(TI i = 0; i < p.num_branches; i++){
                TI rank = p.input_ranks[i];
                TI feature_start = (rank >= 3) ? 2 : (rank >= 2 ? 1 : 0);
                TI flat = 1;
                for(TI d = feature_start; d < rank; d++) flat *= p.input_shapes[i][d];
                arr.set((unsigned)i, (int)flat);
            }
            if(p.num_branches > 0) return arr;
        }
        arr.set(0u, (int)input_dim);
        return arr;
    }

    int create_state() {
        rlt::dyn::State<TI> state;
        state.batch_size = 1;
        state.layer = &model;
        rlt::malloc(device, state);
        states.push_back(std::move(state));
        return (int)(states.size() - 1);
    }

    void reset_state(int id) {
        if(id < 0 || id >= (int)states.size()) return;
        rlt::reset(device, model, states[id]);
    }

    emscripten::val evaluate_step(int state_id, emscripten::val js_input) {
        unsigned int len = js_input["length"].as<unsigned int>();
        std::vector<float> input_data(len);
        emscripten::val heap_view = emscripten::val(emscripten::typed_memory_view(len, input_data.data()));
        heap_view.call<void>("set", js_input);

        rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> input_tensor;
        TI input_shape[] = {(TI)1, (TI)len};
        rlt::dyn::set_shape(input_tensor, (TI)2, input_shape);
        input_tensor.type = rlt::dyn::Type::FLOAT32;
        input_tensor.data = input_data.data();

        rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> output_tensor;
        TI output_shape[] = {(TI)model.output_size};
        rlt::dyn::set_shape(output_tensor, (TI)1, output_shape);
        output_tensor.type = rlt::dyn::Type::FLOAT32;
        output_tensor.data = output_cache.data();
        output_tensor.capacity = output_cache.size();

        if(state_id >= 0 && state_id < (int)states.size()){
            rlt::evaluate_step(device, model, input_tensor, states[state_id], output_tensor, buffer);
        } else {
            rlt::evaluate(device, model, input_tensor, output_tensor, buffer);
        }

        return emscripten::val(emscripten::typed_memory_view(output_dim, output_cache.data()));
    }

    emscripten::val evaluate(emscripten::val js_input) {
        return evaluate_step(-1, js_input);
    }

    emscripten::val evaluate_tuple(emscripten::val js_inputs) {
        unsigned int n = js_inputs["length"].as<unsigned int>();
        if(n > rlt::dyn::TensorTuple<TI>::MAX_TENSORS) n = rlt::dyn::TensorTuple<TI>::MAX_TENSORS;

        const rlt::dyn::layers::Parallel<TI>* parallel = nullptr;
        if(model.type == rlt::dyn::LayerType::PARALLEL && model.data){
            parallel = &model.template as<const rlt::dyn::layers::Parallel<TI>>();
        }

        std::vector<std::vector<float>> storage(n);
        rlt::dyn::TensorTuple<TI> tuple;
        tuple.num_tensors = n;
        for(unsigned int i = 0; i < n; i++){
            emscripten::val js_in = js_inputs[i];
            unsigned int len = js_in["length"].as<unsigned int>();
            storage[i].resize(len);
            emscripten::val heap_view = emscripten::val(emscripten::typed_memory_view(len, storage[i].data()));
            heap_view.call<void>("set", js_in);

            TI shape[rlt::dyn::TensorSpecification<TI>::MAX_RANK]; TI rank;
            if(parallel != nullptr && i < parallel->num_branches && parallel->input_ranks[i] > 0){
                // Use the per-branch INPUT_SHAPE that the parallel persist saved at checkpoint-write time.
                // This is the only source that handles non-square spatial dimensions; the previous
                // sqrt(spatial) heuristic silently failed for aspects like 80x50.
                rank = parallel->input_ranks[i];
                for(TI d = 0; d < rank; d++) shape[d] = parallel->input_shapes[i][d];
                // Stored shape is [STEPS, BATCH, ...features]. Override the leading step/batch dims to 1
                // so the product matches the single-sample flat input provided by JS.
                TI leading = (rank >= 3) ? 2 : (rank >= 2 ? 1 : 0);
                for(TI d = 0; d < leading; d++) shape[d] = 1;
            } else {
                shape[0] = 1; shape[1] = len; rank = 2;
            }
            rlt::dyn::set_shape(tuple.tensors[i], rank, shape);
            tuple.tensors[i].type = rlt::dyn::Type::FLOAT32;
            tuple.tensors[i].data = storage[i].data();
            tuple.tensors[i].capacity = len;
        }

        rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> output_tensor;
        TI output_shape[] = {(TI)model.output_size};
        rlt::dyn::set_shape(output_tensor, (TI)1, output_shape);
        output_tensor.type = rlt::dyn::Type::FLOAT32;
        output_tensor.data = output_cache.data();
        output_tensor.capacity = output_cache.size();

        rlt::evaluate(device, model, tuple, output_tensor, buffer);
        return emscripten::val(emscripten::typed_memory_view(output_dim, output_cache.data()));
    }

    emscripten::val verify() {
        if(example_inputs_data.empty() || example_output_data.empty()){
            emscripten::val ret = emscripten::val::object();
            ret.set("pass", false); ret.set("error", std::string("no example data")); return ret;
        }
        propagate_example_shapes();
        rlt::dyn::Buffer<TI> verify_buffer;
        verify_buffer.layer = &model;
        rlt::malloc(device, verify_buffer);

        std::vector<float> verify_output(model.output_size);
        rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> output_tensor;
        TI out_shape[] = {(TI)model.output_size};
        rlt::dyn::set_shape(output_tensor, (TI)1, out_shape);
        output_tensor.type = rlt::dyn::Type::FLOAT32;
        output_tensor.data = verify_output.data();
        output_tensor.capacity = verify_output.size();
        size_t actual_n = model.output_size;

        if(model.type == rlt::dyn::LayerType::PARALLEL){
            rlt::dyn::TensorTuple<TI> tuple;
            TI n = (TI)example_inputs_data.size();
            if(n > rlt::dyn::TensorTuple<TI>::MAX_TENSORS) n = rlt::dyn::TensorTuple<TI>::MAX_TENSORS;
            tuple.num_tensors = n;
            for(TI i = 0; i < n; i++){
                rlt::dyn::set_shape(tuple.tensors[i],
                    (TI)example_inputs_shape[i].size(),
                    example_inputs_shape[i].data());
                tuple.tensors[i].type = rlt::dyn::Type::FLOAT32;
                tuple.tensors[i].data = example_inputs_data[i].data();
                tuple.tensors[i].capacity = example_inputs_data[i].size();
            }
            rlt::evaluate(device, model, tuple, output_tensor, verify_buffer);
        }
        else{
            rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> input_tensor;
            rlt::dyn::set_shape(input_tensor,
                (TI)example_inputs_shape[0].size(),
                example_inputs_shape[0].data());
            input_tensor.type = rlt::dyn::Type::FLOAT32;
            input_tensor.data = example_inputs_data[0].data();
            input_tensor.capacity = example_inputs_data[0].size();
            rlt::evaluate(device, model, input_tensor, output_tensor, verify_buffer);
        }

        float max_diff = 0;
        size_t compare_n = example_output_data.size() < actual_n ? example_output_data.size() : actual_n;
        for(size_t i = 0; i < compare_n; i++){
            float diff = std::abs(verify_output[i] - example_output_data[i]);
            if(diff > max_diff) max_diff = diff;
        }
        rlt::free(device, verify_buffer);
        propagate_runtime_shapes();
        emscripten::val ret = emscripten::val::object();
        ret.set("max_diff", max_diff);
        ret.set("pass", max_diff < 1e-4f);
        if(max_diff >= 1e-4f){
            emscripten::val expected = emscripten::val::array();
            emscripten::val actual = emscripten::val::array();
            for(size_t i = 0; i < compare_n; i++){
                expected.set((unsigned)i, example_output_data[i]);
                actual.set((unsigned)i, verify_output[i]);
            }
            ret.set("expected", expected);
            ret.set("actual", actual);
        }
        return ret;
    }

    void destroy() {
        if(!loaded) return;
        for(auto& s : states) rlt::free(device, s);
        states.clear();
        rlt::free(device, buffer);
        rlt::free(device, model);
        output_cache.clear();
        example_inputs_data.clear();
        example_inputs_shape.clear();
        example_output_data.clear();
        example_output_shape.clear();
        loaded = false;
        input_dim = 0;
        output_dim = 0;
    }

    ~DynInference() { destroy(); }
};

EMSCRIPTEN_BINDINGS(dyn_inference_module) {
    emscripten::class_<DynInference>("DynInference")
        .constructor<>()
        .function("load", &DynInference::load)
        .function("get_checkpoint_name", &DynInference::get_checkpoint_name)
        .function("get_meta", &DynInference::get_meta)
        .function("get_input_dim", &DynInference::get_input_dim)
        .function("get_output_dim", &DynInference::get_output_dim)
        .function("get_num_branches", &DynInference::get_num_branches)
        .function("get_input_dims", &DynInference::get_input_dims)
        .function("create_state", &DynInference::create_state)
        .function("reset_state", &DynInference::reset_state)
        .function("evaluate_step", &DynInference::evaluate_step)
        .function("evaluate", &DynInference::evaluate)
        .function("evaluate_tuple", &DynInference::evaluate_tuple)
        .function("verify", &DynInference::verify)
        .function("destroy", &DynInference::destroy);
}

#include <rl_tools/operations/cpu.h>
#include <rl_tools/persist/backends/hdf5/hdf5.h>
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#include <rl_tools/dyn/persist.h>

#include <cstdio>
#include <cmath>

namespace rlt = rl_tools;

int main(int argc, char** argv){
    const char* path = (argc > 1) ? argv[1] : "tests/data/test_dyn_wasm_checkpoint.h5";

    using DEVICE = rlt::devices::DefaultCPU;
    using TI = typename DEVICE::index_t;
    DEVICE device;

    rlt::persist::backends::hdf5::File file(path, rlt::persist::backends::hdf5::Mode::READ);

    bool has_actor = H5Lexists(file.id, "actor", H5P_DEFAULT) > 0;
    auto model_group = rlt::get_group(device, file, has_actor ? "actor" : "model");
    rlt::dyn::Layer<TI> model;
    if(!rlt::load(device, model, model_group)){
        printf("ERROR: failed to load model\n");
        return 1;
    }
    printf("Model loaded (type=%d, children=%lu)\n", (int)model.type, (unsigned long)model.num_children);

    bool has_example = H5Lexists(file.id, "example", H5P_DEFAULT) > 0;
    const char* input_name = has_example ? "example/input" : "test_input";
    const char* output_name = has_example ? "example/output" : "expected_output";

    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> input_tensor;
    {
        hid_t ds = H5Dopen2(file.id, input_name, H5P_DEFAULT);
        hid_t space = H5Dget_space(ds);
        int rank = H5Sget_simple_extent_ndims(space);
        hsize_t dims[8];
        H5Sget_simple_extent_dims(space, dims, nullptr);
        TI input_shape[8] = {};
        TI input_size = 1;
        for(int i = 0; i < rank; i++){ input_shape[i] = (TI)dims[i]; input_size *= (TI)dims[i]; }
        rlt::dyn::set_shape(input_tensor, (TI)rank, input_shape);
        input_tensor.type = rlt::dyn::Type::FLOAT32;
        rlt::malloc(device, input_tensor);
        H5Dread(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, input_tensor.data);
        H5Sclose(space);
        H5Dclose(ds);
        printf("Input: rank=%d, size=%lu\n", rank, (unsigned long)input_size);
    }

    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> expected;
    {
        hid_t ds = H5Dopen2(file.id, output_name, H5P_DEFAULT);
        hid_t space = H5Dget_space(ds);
        int out_rank = H5Sget_simple_extent_ndims(space);
        hsize_t out_dims[8];
        H5Sget_simple_extent_dims(space, out_dims, nullptr);
        TI n_total = 1;
        for(int d = 0; d < out_rank; d++) n_total *= (TI)out_dims[d];
        TI expected_shape[] = {n_total};
        rlt::dyn::set_shape(expected, (TI)1, expected_shape);
        expected.type = rlt::dyn::Type::FLOAT32;
        rlt::malloc(device, expected);
        H5Dread(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, expected.data);
        H5Sclose(space);
        H5Dclose(ds);
    }

    rlt::dyn::propagate_shapes(model, input_tensor.shape, input_tensor.rank);
    rlt::dyn::Buffer<TI> buffer;
    buffer.layer = &model;
    rlt::malloc(device, buffer);

    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> output;
    TI output_shape[] = {model.output_size};
    rlt::dyn::set_shape(output, (TI)1, output_shape);
    output.type = rlt::dyn::Type::FLOAT32;
    rlt::malloc(device, output);

    bool ok = rlt::evaluate(device, model, input_tensor, output, buffer);
    if(!ok){ printf("ERROR: evaluate failed\n"); return 1; }

    float max_diff = 0;
    TI output_size = expected.size();
    printf("Output size: %lu\n", (unsigned long)output_size);
    TI print_n = output_size < 10 ? output_size : 10;
    for(TI i = 0; i < print_n; i++){
        float got = rlt::get(device, output, i);
        float exp = rlt::get(device, expected, i);
        printf("  [%lu] got=%e expected=%e\n", (unsigned long)i, got, exp);
    }
    for(TI i = 0; i < output_size; i++){
        float got = rlt::get(device, output, i);
        float exp = rlt::get(device, expected, i);
        float diff = std::fabs(got - exp);
        if(diff > max_diff) max_diff = diff;
    }

    printf("Max absolute difference: %e\n", max_diff);
    if(max_diff < 1e-5f){
        printf("PASS\n");
        return 0;
    } else {
        printf("FAIL (threshold: 1e-5)\n");
        TI print_n = output_size < 8 ? output_size : 8;
        for(TI i = 0; i < print_n; i++){
            printf("  [%lu] got=%f expected=%f\n", (unsigned long)i, rlt::get(device, output, i), rlt::get(device, expected, i));
        }
        return 1;
    }
}

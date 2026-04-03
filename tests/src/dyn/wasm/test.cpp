#include <rl_tools/operations/cpu.h>
#include <rl_tools/persist/backends/h5/h5.h>
#include <rl_tools/persist/backends/h5/operations_generic.h>
#include <rl_tools/dyn/persist.h>

#include <cstdio>
#include <cmath>
#include <hdf5.h>

namespace rlt = rl_tools;

int main(int argc, char** argv){
    const char* path = (argc > 1) ? argv[1] : "test_checkpoint.h5";

    using DEVICE = rlt::devices::DefaultCPU;
    using TI = typename DEVICE::index_t;
    DEVICE device;

    // Open HDF5 file
    hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if(file < 0){ printf("ERROR: cannot open %s\n", path); return 1; }

    // Read test metadata
    char input_dim_str[16], batch_size_str[16];
    rlt::persist::backends::h5::detail::read_string_attribute(file, "input_dim", input_dim_str, sizeof(input_dim_str));
    rlt::persist::backends::h5::detail::read_string_attribute(file, "batch_size", batch_size_str, sizeof(batch_size_str));
    TI input_dim = 0, batch_size = 0;
    for(int i = 0; input_dim_str[i] >= '0' && input_dim_str[i] <= '9'; i++) input_dim = input_dim * 10 + (input_dim_str[i] - '0');
    for(int i = 0; batch_size_str[i] >= '0' && batch_size_str[i] <= '9'; i++) batch_size = batch_size * 10 + (batch_size_str[i] - '0');
    printf("Input dim: %lu, batch size: %lu\n", (unsigned long)input_dim, (unsigned long)batch_size);

    // Load model
    auto model_group = rlt::get_group(device, file, "model");
    rlt::dyn::Layer<TI> model;
    if(!rlt::load(device, model, model_group)){
        printf("ERROR: failed to load model\n");
        H5Fclose(file);
        return 1;
    }
    printf("Model loaded (type=%d, children=%lu)\n", (int)model.type, (unsigned long)model.num_children);

    // Prepare input tensor
    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> input_tensor;
    TI input_shape[] = {batch_size, input_dim};
    rlt::dyn::set_shape(input_tensor, (TI)2, input_shape);
    input_tensor.type = rlt::dyn::Type::FLOAT32;
    rlt::malloc(device, input_tensor);

    // Read test input directly from HDF5
    {
        hid_t ds = H5Dopen2(file, "test_input", H5P_DEFAULT);
        H5Dread(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, input_tensor.data);
        H5Dclose(ds);
    }

    // Read expected output
    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> expected;
    {
        hid_t ds = H5Dopen2(file, "expected_output", H5P_DEFAULT);
        hid_t space = H5Dget_space(ds);
        hsize_t n;
        H5Sget_simple_extent_dims(space, &n, nullptr);
        TI expected_shape[] = {(TI)n};
        rlt::dyn::set_shape(expected, (TI)1, expected_shape);
        expected.type = rlt::dyn::Type::FLOAT32;
        rlt::malloc(device, expected);
        H5Dread(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, expected.data);
        H5Sclose(space);
        H5Dclose(ds);
    }

    H5Fclose(file);

    // Propagate shapes and allocate buffer
    rlt::dyn::propagate_shapes(model, input_shape, (TI)2, batch_size * input_dim);
    rlt::dyn::Buffer<TI> buffer;
    buffer.layer = &model;
    rlt::malloc(device, buffer);

    // Forward pass
    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> output;
    TI output_shape[] = {model.output_size};
    rlt::dyn::set_shape(output, (TI)1, output_shape);
    output.type = rlt::dyn::Type::FLOAT32;
    rlt::malloc(device, output);

    bool ok = rlt::evaluate(device, model, input_tensor, output, buffer);
    if(!ok){ printf("ERROR: evaluate failed\n"); return 1; }

    // Compare
    float max_diff = 0;
    TI output_size = expected.size;
    printf("Output size: %lu\n", (unsigned long)output_size);
    for(TI i = 0; i < output_size; i++){
        float got = rlt::dyn::get(device, output, i);
        float exp = rlt::dyn::get(device, expected, i);
        float diff = std::fabs(got - exp);
        if(diff > max_diff) max_diff = diff;
    }

    printf("Max absolute difference: %e\n", max_diff);
    if(max_diff < 1e-5f){
        printf("PASS\n");
        return 0;
    } else {
        printf("FAIL (threshold: 1e-5)\n");
        // Print first few values for debugging
        TI print_n = output_size < 8 ? output_size : 8;
        for(TI i = 0; i < print_n; i++){
            printf("  [%lu] got=%f expected=%f\n", (unsigned long)i, rlt::dyn::get(device, output, i), rlt::dyn::get(device, expected, i));
        }
        return 1;
    }
}

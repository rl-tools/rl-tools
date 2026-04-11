#include <rl_tools/operations/cpu.h>
#include <rl_tools/persist/backends/hdf5/hdf5.h>
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#include <rl_tools/dyn/persist.h>

#include <cstdio>
#include <cmath>

namespace rlt = rl_tools;

int main(int argc, char** argv){
    if(argc < 2){
        printf("Usage: %s <checkpoint.h5>\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];

    using DEVICE = rlt::devices::DefaultCPU;
    using TI = typename DEVICE::index_t;
    DEVICE device;

    rlt::persist::backends::hdf5::File file(path, rlt::persist::backends::hdf5::Mode::READ);

    auto actor_group = rlt::get_group(device, file, "actor");
    rlt::dyn::Layer<TI> model;
    if(!rlt::load(device, model, actor_group)){
        printf("ERROR: failed to load model from 'actor' group\n");
        return 1;
    }
    printf("Model loaded (type=%d, children=%lu)\n", (int)model.type, (unsigned long)model.num_children);

    auto example_group = rlt::get_group(device, file, "example");

    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> input;
    if(!rlt::load(device, input, example_group, "input")){
        printf("ERROR: failed to load example/input\n");
        return 1;
    }
    printf("Input: rank=%u, size=%lu, shape=[", (unsigned)input.rank, (unsigned long)input.size());
    for(TI d = 0; d < input.rank; d++) printf("%s%lu", d ? "," : "", (unsigned long)input.shape[d]);
    printf("]\n");

    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> expected;
    if(!rlt::load(device, expected, example_group, "output")){
        printf("ERROR: failed to load example/output\n");
        return 1;
    }
    printf("Expected output: rank=%u, size=%lu\n", (unsigned)expected.rank, (unsigned long)expected.size());

    rlt::dyn::propagate_shapes(model, input.shape, input.rank);

    rlt::dyn::Buffer<TI> buffer;
    buffer.layer = &model;
    rlt::malloc(device, buffer);

    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> output;
    TI output_shape[] = {model.output_size};
    rlt::dyn::set_shape(output, (TI)1, output_shape);
    output.type = rlt::dyn::Type::FLOAT32;
    rlt::malloc(device, output);

    if(!rlt::evaluate(device, model, input, output, buffer)){
        printf("ERROR: evaluate failed\n");
        return 1;
    }

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
        return 1;
    }
}

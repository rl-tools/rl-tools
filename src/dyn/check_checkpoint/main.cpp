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
    auto inputs_group = rlt::get_group(device, example_group, "inputs");
    auto outputs_group = rlt::get_group(device, example_group, "outputs");

    rlt::dyn::TensorTuple<TI> tuple_input;
    tuple_input.num_tensors = 0;
    for(TI i = 0; i < rlt::dyn::TensorTuple<TI>::MAX_TENSORS; i++){
        char name[4] = {(char)('0' + i), '\0', '\0', '\0'};
        if(!rlt::load(device, tuple_input.tensors[i], inputs_group, name)){
            break;
        }
        tuple_input.num_tensors = i + 1;
    }
    if(tuple_input.num_tensors == 0){
        printf("ERROR: no example inputs found\n");
        return 1;
    }
    for(TI i = 0; i < tuple_input.num_tensors; i++){
        auto& t = tuple_input.tensors[i];
        printf("Input %lu: rank=%u, size=%lu, shape=[", (unsigned long)i, (unsigned)t.rank, (unsigned long)t.size());
        for(TI d = 0; d < t.rank; d++) printf("%s%lu", d ? "," : "", (unsigned long)t.shape[d]);
        printf("]\n");
    }

    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> expected;
    if(!rlt::load(device, expected, outputs_group, "0")){
        printf("ERROR: failed to load example/outputs/0\n");
        return 1;
    }
    printf("Expected output: rank=%u, size=%lu\n", (unsigned)expected.rank, (unsigned long)expected.size());

    rlt::dyn::propagate_shapes(model, tuple_input);

    rlt::dyn::Buffer<TI> buffer;
    buffer.layer = &model;
    rlt::malloc(device, buffer);

    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> output;
    TI output_shape[] = {model.output_size};
    rlt::dyn::set_shape(output, (TI)1, output_shape);
    output.type = rlt::dyn::Type::FLOAT32;
    rlt::malloc(device, output);

    if(!rlt::evaluate(device, model, tuple_input, output, buffer)){
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

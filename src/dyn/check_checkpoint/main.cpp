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

    // For parallel models, split the flat example input into a tuple with per-branch shapes.
    // The checkpoint stores input_dim_0/1 attributes that tell us the flat split sizes.
    // Branch shapes are inferred: if a branch starts with Conv2D, reshape to [batch, H, W, C].
    bool is_parallel = (model.type == rlt::dyn::LayerType::PARALLEL && model.num_children >= 2);
    rlt::dyn::TensorTuple<TI> tuple_input;
    if(is_parallel){
        auto ag2 = rlt::get_group(device, file, "actor");
        TI dim_a = 0, dim_b = 0;
        if(rlt::attribute_exists(device, ag2, "input_dim_0")) dim_a = rlt::get_attribute_int<TI>(device, ag2, "input_dim_0");
        if(rlt::attribute_exists(device, ag2, "input_dim_1")) dim_b = rlt::get_attribute_int<TI>(device, ag2, "input_dim_1");
        if(dim_a == 0 || dim_b == 0 || dim_a + dim_b != input.size()){
            printf("ERROR: cannot determine parallel input split (dim_a=%lu dim_b=%lu input_size=%lu)\n",
                (unsigned long)dim_a, (unsigned long)dim_b, (unsigned long)input.size());
            return 1;
        }
        tuple_input.num_tensors = 2;
        // Branch 0: check if it starts with conv2d for spatial reshape
        auto& branch_0 = model.children[0];
        auto* first_layer = &branch_0;
        while(first_layer->num_children > 0 && (first_layer->type == rlt::dyn::LayerType::SEQUENTIAL || first_layer->type == rlt::dyn::LayerType::MLP))
            first_layer = &first_layer->children[0];
        if(first_layer->type == rlt::dyn::LayerType::CONV2D){
            auto& conv = first_layer->template as<const rlt::dyn::layers::Conv2d<TI>>();
            TI ic = conv.input_channels;
            TI spatial = dim_a / ic;
            TI side = 1; while(side * side < spatial) side++;
            TI shape_a[] = {(TI)1, side, side, ic};
            rlt::dyn::set_shape(tuple_input.tensors[0], (TI)4, shape_a);
        } else {
            TI shape_a[] = {(TI)1, dim_a};
            rlt::dyn::set_shape(tuple_input.tensors[0], (TI)2, shape_a);
        }
        tuple_input.tensors[0].type = rlt::dyn::Type::FLOAT32;
        tuple_input.tensors[0].data = input.data;
        tuple_input.tensors[0].capacity = dim_a;
        TI shape_b[] = {(TI)1, dim_b};
        rlt::dyn::set_shape(tuple_input.tensors[1], (TI)2, shape_b);
        tuple_input.tensors[1].type = rlt::dyn::Type::FLOAT32;
        tuple_input.tensors[1].data = reinterpret_cast<char*>(input.data) + dim_a * sizeof(float);
        tuple_input.tensors[1].capacity = dim_b;
        rlt::dyn::propagate_shapes(model, tuple_input);
    } else {
        rlt::dyn::propagate_shapes(model, input.shape, input.rank);
    }

    rlt::dyn::Buffer<TI> buffer;
    buffer.layer = &model;
    rlt::malloc(device, buffer);

    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> output;
    TI output_shape[] = {model.output_size};
    rlt::dyn::set_shape(output, (TI)1, output_shape);
    output.type = rlt::dyn::Type::FLOAT32;
    rlt::malloc(device, output);

    bool eval_ok = is_parallel
        ? rlt::evaluate(device, model, tuple_input, output, buffer)
        : rlt::evaluate(device, model, input, output, buffer);
    if(!eval_ok){
        printf("ERROR: evaluate failed\n");
        return 1;
    }

    float max_diff = 0;
    TI output_size = expected.size();
    printf("Output size: %lu\n", (unsigned long)output_size);
    TI print_n = output_size < 10 ? output_size : 10;
    for(TI i = 0; i < print_n; i++){
        float got = rlt::dyn::get(device, output, i);
        float exp = rlt::dyn::get(device, expected, i);
        printf("  [%lu] got=%e expected=%e\n", (unsigned long)i, got, exp);
    }
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
        return 1;
    }
}

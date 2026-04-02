#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_DYN_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_DYN_PERSIST_H

#include "model.h"
#include "operations_generic.h"
#include "../utils/string/operations_generic.h"
#include "../persist/backends/tar/operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace dyn::persist_helpers{
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT ActivationFunction parse_activation_function(DEVICE& device, const char* name, TI len){
            if(utils::string::compare(name, "IDENTITY", 8)) return ActivationFunction::IDENTITY;
            if(utils::string::compare(name, "RELU", 4)) return ActivationFunction::RELU;
            if(utils::string::compare(name, "relu", 4)) return ActivationFunction::RELU;
            if(utils::string::compare(name, "GELU", 4)) return ActivationFunction::GELU;
            if(utils::string::compare(name, "gelu", 4)) return ActivationFunction::GELU;
            if(utils::string::compare(name, "TANH", 4)) return ActivationFunction::TANH;
            if(utils::string::compare(name, "tanh", 4)) return ActivationFunction::TANH;
            if(utils::string::compare(name, "FAST_TANH", 9)) return ActivationFunction::FAST_TANH;
            if(utils::string::compare(name, "fast_tanh", 9)) return ActivationFunction::FAST_TANH;
            if(utils::string::compare(name, "SIGMOID", 7)) return ActivationFunction::SIGMOID;
            if(utils::string::compare(name, "sigmoid", 7)) return ActivationFunction::SIGMOID;
            return ActivationFunction::IDENTITY;
        }

        template <typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT dyn::Type parse_dtype(const char* name, TI len){
            if(utils::string::compare(name, "float32", 7)) return dyn::Type::FLOAT32;
            if(utils::string::compare(name, "float64", 7)) return dyn::Type::FLOAT64;
            if(utils::string::compare(name, "bf16", 4)) return dyn::Type::BF16;
            if(utils::string::compare(name, "int8", 4)) return dyn::Type::INT8;
            return dyn::Type::FLOAT32;
        }
    }

    // --- Load a dyn::Tensor from a persist group (runtime shape discovery) ---
    template <typename DEVICE, typename TI, typename GROUP>
    RL_TOOLS_FUNCTION_PLACEMENT bool load(DEVICE& device, dyn::Tensor<dyn::TensorSpecification<TI>>& tensor, GROUP& group, const char* name){
        using namespace dyn;
        auto tensor_group = get_group(device, group, name);
        using GROUP_SPEC = typename GROUP::SPEC;
        constexpr TI MAX_PATH = GROUP_SPEC::MAX_PATH_LENGTH;

        char current_path[MAX_PATH];
        utils::string::copy<TI>(current_path, tensor_group.path, MAX_PATH);
        TI path_len = utils::string::length(current_path, MAX_PATH);
        TI sep_pos = path_len;
        if(path_len > 0 && path_len + 1 < MAX_PATH){
            current_path[path_len] = '/';
            current_path[path_len + 1] = '\0';
            sep_pos = path_len + 1;
        }

        // Read metadata
        utils::string::copy(current_path + sep_pos, "meta", MAX_PATH - sep_pos);
        constexpr TI METADATA_SIZE = 200;
        char metadata[METADATA_SIZE];
        TI read_size = 0;
        if(!rl_tools::persist::backends::tar::get(device, tensor_group.data, current_path, metadata, METADATA_SIZE, read_size)){
            return false;
        }
        if(read_size < METADATA_SIZE){
            metadata[read_size] = '\0';
        }

        // Parse dtype
        TI dtype_pos, dtype_len;
        if(rl_tools::persist::backends::tar::seek_in_metadata(device, metadata, read_size, "dtype", dtype_pos, dtype_len)){
            tensor.type = dyn::persist_helpers::parse_dtype(metadata + dtype_pos, dtype_len);
        }

        // Parse num_dims
        TI ndims_pos, ndims_len;
        if(!rl_tools::persist::backends::tar::seek_in_metadata(device, metadata, read_size, "num_dims", ndims_pos, ndims_len)){
            return false;
        }
        tensor.rank = utils::string::string_to_int<TI>(metadata + ndims_pos, ndims_len);

        // Parse each dim
        tensor.size = 1;
        for(TI d = 0; d < tensor.rank; d++){
            char dim_key[16] = "dim_";
            char digit[2] = {static_cast<char>('0' + d), '\0'};
            utils::string::copy(dim_key + 4, digit, 12);
            TI dim_pos, dim_len;
            if(!rl_tools::persist::backends::tar::seek_in_metadata(device, metadata, read_size, dim_key, dim_pos, dim_len)){
                return false;
            }
            tensor.shape[d] = utils::string::string_to_int<TI>(metadata + dim_pos, dim_len);
            tensor.size *= tensor.shape[d];
        }

        // Allocate and read data
        rl_tools::malloc(device, tensor);
        utils::string::copy(current_path + sep_pos, "data", MAX_PATH - sep_pos);
        TI data_size = tensor.size * dyn::size_of<TI>(tensor.type);
        TI data_read_size = 0;
        if(!rl_tools::persist::backends::tar::get(device, tensor_group.data, current_path, reinterpret_cast<char*>(tensor.data), data_size, data_read_size)){
            return false;
        }
        return true;
    }

    // --- Load a dyn::Layer from a persist group ---
    template <typename DEVICE, typename TI, typename GROUP>
    RL_TOOLS_FUNCTION_PLACEMENT bool load(DEVICE& device, dyn::Layer<TI>& layer, GROUP& group){
        using namespace dyn;
        constexpr TI ATTR_BUF_SIZE = 64;
        char type_str[ATTR_BUF_SIZE];
        get_attribute<char*>(device, group, "type", type_str, ATTR_BUF_SIZE);

        if(utils::string::compare(type_str, "dense", 5)){
            layer.type = LayerType::DENSE;
            auto* d = new layers::Dense<TI>();
            auto weights_group = get_group(device, group, "weights");
            load(device, d->weights, weights_group, "parameters");
            auto biases_group = get_group(device, group, "biases");
            load(device, d->biases, biases_group, "parameters");
            d->output_dim = d->weights.shape[0];
            d->input_dim = d->weights.shape[1];
            char af_str[ATTR_BUF_SIZE];
            get_attribute<char*>(device, group, "activation_function", af_str, ATTR_BUF_SIZE);
            d->activation_function = dyn::persist_helpers::parse_activation_function(device, af_str, ATTR_BUF_SIZE);
            layer.data = d;
            return true;
        }
        else if(utils::string::compare(type_str, "gru", 3)){
            layer.type = LayerType::GRU;
            auto* g = new layers::GRU<TI>();
            auto wi_group = get_group(device, group, "weights_input");
            load(device, g->weights_input, wi_group, "parameters");
            auto bi_group = get_group(device, group, "biases_input");
            load(device, g->biases_input, bi_group, "parameters");
            auto wh_group = get_group(device, group, "weights_hidden");
            load(device, g->weights_hidden, wh_group, "parameters");
            auto bh_group = get_group(device, group, "biases_hidden");
            load(device, g->biases_hidden, bh_group, "parameters");
            auto ihs_group = get_group(device, group, "initial_hidden_state");
            load(device, g->initial_hidden_state, ihs_group, "parameters");
            g->hidden_dim = g->weights_input.shape[0] / 3;
            g->input_dim = g->weights_input.shape[1];
            layer.data = g;
            return true;
        }
        else if(utils::string::compare(type_str, "sequential", 10)){
            layer.type = LayerType::SEQUENTIAL;
            auto* seq = new layers::Sequential<TI>();
            auto layers_group = get_group(device, group, "layers");
            // Count layers by probing for groups "0", "1", "2", ...
            TI count = 0;
            for(TI i = 0; i < 100; i++){
                char idx_str[10];
                utils::string::int_to_string<long int, TI>(idx_str, 10, i);
                if(!group_exists(device, layers_group, idx_str)){
                    break;
                }
                count++;
            }
            seq->num_layers = count;
            seq->layers = new Layer<TI>[count];
            for(TI i = 0; i < count; i++){
                char idx_str[10];
                utils::string::int_to_string<long int, TI>(idx_str, 10, i);
                auto layer_group = get_group(device, layers_group, idx_str);
                load(device, seq->layers[i], layer_group);
            }
            layer.data = seq;
            return true;
        }
        else if(utils::string::compare(type_str, "mlp", 3)){
            layer.type = LayerType::MLP;
            auto* mlp = new layers::MLP<TI>();
            auto input_group = get_group(device, group, "input_layer");
            load(device, mlp->input_layer, input_group);
            TI num_layers = get_attribute_int<TI>(device, group, "num_layers");
            mlp->num_hidden_layers = num_layers - 2;
            mlp->hidden_layers = new Layer<TI>[mlp->num_hidden_layers];
            auto hidden_group = get_group(device, group, "hidden_layers");
            for(TI i = 0; i < mlp->num_hidden_layers; i++){
                char idx_str[10];
                utils::string::int_to_string<long int, TI>(idx_str, 10, i);
                auto hl_group = get_group(device, hidden_group, idx_str);
                load(device, mlp->hidden_layers[i], hl_group);
            }
            auto output_group = get_group(device, group, "output_layer");
            load(device, mlp->output_layer, output_group);
            layer.data = mlp;
            return true;
        }
        else if(utils::string::compare(type_str, "sample_and_squash", 17)){
            layer.type = LayerType::SAMPLE_AND_SQUASH;
            layer.data = nullptr;
            return true;
        }
        else if(utils::string::compare(type_str, "standardize", 11)){
            layer.type = LayerType::STANDARDIZE;
            auto* s = new layers::Standardize<TI>();
            auto mean_group = get_group(device, group, "mean");
            load(device, s->mean, mean_group, "parameters");
            auto prec_group = get_group(device, group, "precision");
            load(device, s->precision, prec_group, "parameters");
            s->dim = s->mean.size;
            layer.data = s;
            return true;
        }
        else if(utils::string::compare(type_str, "flatten", 7)){
            layer.type = LayerType::FLATTEN;
            layer.data = nullptr;
            return true;
        }
        else if(utils::string::compare(type_str, "unflatten", 9)){
            layer.type = LayerType::UNFLATTEN;
            layer.data = nullptr;
            return true;
        }
        else if(utils::string::compare(type_str, "avg_pool2d", 10)){
            layer.type = LayerType::AVG_POOL2D;
            layer.data = nullptr;
            return true;
        }
        else if(utils::string::compare(type_str, "max_pool2d", 10)){
            layer.type = LayerType::MAX_POOL2D;
            auto* mp = new layers::MaxPool2d<TI>();
            mp->kernel_height = get_attribute_int<TI>(device, group, "kernel_height");
            mp->kernel_width = get_attribute_int<TI>(device, group, "kernel_width");
            mp->stride_h = get_attribute_int<TI>(device, group, "stride_h");
            mp->stride_w = get_attribute_int<TI>(device, group, "stride_w");
            mp->padding_h = get_attribute_int<TI>(device, group, "padding_h");
            mp->padding_w = get_attribute_int<TI>(device, group, "padding_w");
            layer.data = mp;
            return true;
        }
        else if(utils::string::compare(type_str, "conv2d", 6)){
            layer.type = LayerType::CONV2D;
            auto* c = new layers::Conv2d<TI>();
            auto weights_group = get_group(device, group, "weights");
            load(device, c->weights, weights_group, "parameters");
            auto biases_group = get_group(device, group, "biases");
            load(device, c->biases, biases_group, "parameters");
            c->output_channels = get_attribute_int<TI>(device, group, "output_channels");
            c->input_channels = get_attribute_int<TI>(device, group, "input_channels");
            c->kernel_height = get_attribute_int<TI>(device, group, "kernel_height");
            c->kernel_width = get_attribute_int<TI>(device, group, "kernel_width");
            c->stride_h = get_attribute_int<TI>(device, group, "stride_h");
            c->stride_w = get_attribute_int<TI>(device, group, "stride_w");
            c->padding_h = get_attribute_int<TI>(device, group, "padding_h");
            c->padding_w = get_attribute_int<TI>(device, group, "padding_w");
            char af_str[ATTR_BUF_SIZE];
            get_attribute<char*>(device, group, "activation_function", af_str, ATTR_BUF_SIZE);
            c->activation_function = dyn::persist_helpers::parse_activation_function(device, af_str, ATTR_BUF_SIZE);
            char norm_str[ATTR_BUF_SIZE];
            get_attribute<char*>(device, group, "normalization", norm_str, ATTR_BUF_SIZE);
            if(utils::string::compare(norm_str, "BATCH_NORM", 10)){
                c->normalization = layers::Conv2d<TI>::Normalization::BATCH_NORM;
            }
            else if(utils::string::compare(norm_str, "LAYER_NORM", 10)){
                c->normalization = layers::Conv2d<TI>::Normalization::LAYER_NORM;
            }
            else{
                c->normalization = layers::Conv2d<TI>::Normalization::NONE;
            }
            if(c->normalization != layers::Conv2d<TI>::Normalization::NONE){
                auto gamma_group = get_group(device, group, "gamma");
                load(device, c->gamma, gamma_group, "parameters");
                auto beta_group = get_group(device, group, "beta");
                load(device, c->beta, beta_group, "parameters");
                if(c->normalization == layers::Conv2d<TI>::Normalization::BATCH_NORM){
                    load(device, c->running_mean, group, "running_mean");
                    load(device, c->running_var, group, "running_var");
                }
            }
            layer.data = c;
            return true;
        }
        else if(utils::string::compare(type_str, "embedding", 9)){
            layer.type = LayerType::EMBEDDING;
            auto* e = new layers::Embedding<TI>();
            auto weights_group = get_group(device, group, "weights");
            load(device, e->weights, weights_group, "parameters");
            e->num_classes = e->weights.shape[0];
            e->embedding_dim = e->weights.shape[1];
            layer.data = e;
            return true;
        }
        else if(utils::string::compare(type_str, "parallel", 8)){
            layer.type = LayerType::PARALLEL;
            auto* p = new layers::Parallel<TI>();
            p->pipeline_a = new Layer<TI>();
            auto pa_group = get_group(device, group, "pipeline_a");
            load(device, *p->pipeline_a, pa_group);
            p->pipeline_b = new Layer<TI>();
            auto pb_group = get_group(device, group, "pipeline_b");
            load(device, *p->pipeline_b, pb_group);
            if(group_exists(device, group, "head")){
                p->head = new Layer<TI>();
                auto head_group = get_group(device, group, "head");
                load(device, *p->head, head_group);
            }
            layer.data = p;
            return true;
        }
        return false;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif

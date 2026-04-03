#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_DYN_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_DYN_PERSIST_H

#include "model.h"
#include "operations_generic.h"
#include "../utils/string/operations_generic.h"

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

    // --- TAR backend: Load dyn::Tensor ---
#ifdef RL_TOOLS_PERSIST_BACKENDS_TAR_OPERATIONS_GENERIC
    template <typename DEVICE, typename TI, typename GROUP_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT bool load(DEVICE& device, dyn::Tensor<dyn::TensorSpecification<TI>>& tensor, persist::backends::tar::ReaderGroup<GROUP_SPEC>& group, const char* name){
        auto tensor_group = get_group(device, group, name);
        constexpr TI MAX_PATH = GROUP_SPEC::MAX_PATH_LENGTH;
        char current_path[MAX_PATH];
        utils::string::copy<TI>(current_path, tensor_group.path, MAX_PATH);
        TI path_len = utils::string::length(current_path, MAX_PATH);
        TI sep_pos = path_len;
        if(path_len > 0 && path_len + 1 < MAX_PATH){ current_path[path_len] = '/'; current_path[path_len+1] = '\0'; sep_pos = path_len+1; }
        utils::string::copy(current_path + sep_pos, "meta", MAX_PATH - sep_pos);
        constexpr TI METADATA_SIZE = 200;
        char metadata[METADATA_SIZE];
        TI read_size = 0;
        if(!persist::backends::tar::get(device, tensor_group.data, current_path, metadata, METADATA_SIZE, read_size)) return false;
        if(read_size < METADATA_SIZE) metadata[read_size] = '\0';
        TI pos, len;
        if(persist::backends::tar::seek_in_metadata(device, metadata, read_size, "dtype", pos, len)) tensor.type = dyn::persist_helpers::parse_dtype(metadata+pos, len);
        if(!persist::backends::tar::seek_in_metadata(device, metadata, read_size, "num_dims", pos, len)) return false;
        tensor.rank = utils::string::string_to_int<TI>(metadata+pos, len);
        tensor.size = 1;
        for(TI d = 0; d < tensor.rank; d++){
            char key[16] = "dim_"; char digit[2] = {static_cast<char>('0'+d), '\0'}; utils::string::copy(key+4, digit, 12);
            if(!persist::backends::tar::seek_in_metadata(device, metadata, read_size, key, pos, len)) return false;
            tensor.shape[d] = utils::string::string_to_int<TI>(metadata+pos, len); tensor.size *= tensor.shape[d];
        }
        rl_tools::malloc(device, tensor);
        utils::string::copy(current_path+sep_pos, "data", MAX_PATH-sep_pos);
        TI data_read_size = 0;
        return persist::backends::tar::get(device, tensor_group.data, current_path, reinterpret_cast<char*>(tensor.data), tensor.size * dyn::size_of<TI>(tensor.type), data_read_size);
    }
#endif

    // --- HDF5 backend: Load dyn::Tensor ---
#ifdef RL_TOOLS_PERSIST_BACKENDS_HDF5_HDF5
    template <typename DEVICE, typename TI, typename GROUP_SPEC>
    bool load(DEVICE& device, dyn::Tensor<dyn::TensorSpecification<TI>>& tensor, const persist::backends::hdf5::Group<GROUP_SPEC>& group, const char* name){
        auto dataset = group.group.getDataSet(name);
        auto dims = dataset.getDimensions();
        tensor.rank = dims.size(); tensor.size = 1;
        for(TI d = 0; d < tensor.rank; d++){ tensor.shape[d] = dims[d]; tensor.size *= dims[d]; }
        auto dt = dataset.getDataType();
        auto dtc = dt.getClass(); auto dts = dt.getSize();
        if(dtc == HighFive::DataTypeClass::Float){ tensor.type = (dts == 8) ? dyn::Type::FLOAT64 : (dts == 2) ? dyn::Type::BF16 : dyn::Type::FLOAT32; }
        else if(dtc == HighFive::DataTypeClass::Integer && dts == 1){ tensor.type = dyn::Type::INT8; }
        rl_tools::malloc(device, tensor);
        if(dtc == HighFive::DataTypeClass::Float && dts == 4){ std::vector<float> buf(tensor.size); dataset.read(buf.data()); for(TI i = 0; i < tensor.size; i++) dyn::set(device, tensor, i, buf[i]); }
        else if(dtc == HighFive::DataTypeClass::Float && dts == 8){ std::vector<double> buf(tensor.size); dataset.read(buf.data()); for(TI i = 0; i < tensor.size; i++) dyn::set(device, tensor, i, static_cast<float>(buf[i])); }
        else{ std::vector<float> buf(tensor.size); dataset.read(buf.data()); for(TI i = 0; i < tensor.size; i++) dyn::set(device, tensor, i, buf[i]); }
        return true;
    }
#endif

    // --- Load a dyn::Layer (backend-agnostic) ---
    template <typename DEVICE, typename TI, typename GROUP>
    RL_TOOLS_FUNCTION_PLACEMENT bool load(DEVICE& device, dyn::Layer<TI>& layer, GROUP& group){
        using namespace dyn;
        constexpr TI ATTR_SIZE = 64;
        char type_str[ATTR_SIZE];
        get_attribute<char*>(device, group, "type", type_str, ATTR_SIZE);

        if(utils::string::compare(type_str, "dense", 5)){
            layer.type = LayerType::DENSE;
            auto* d = new layers::Dense<TI>();
            auto wg = get_group(device, group, "weights"); load(device, d->weights, wg, "parameters");
            auto bg = get_group(device, group, "biases"); load(device, d->biases, bg, "parameters");
            d->output_dim = d->weights.shape[0]; d->input_dim = d->weights.shape[1];
            char af[ATTR_SIZE]; get_attribute<char*>(device, group, "activation_function", af, ATTR_SIZE);
            d->activation_function = persist_helpers::parse_activation_function(device, af, ATTR_SIZE);
            layer.data = d;
        }
        else if(utils::string::compare(type_str, "gru", 3)){
            layer.type = LayerType::GRU;
            auto* g = new layers::GRU<TI>();
            auto wig = get_group(device, group, "weights_input"); load(device, g->weights_input, wig, "parameters");
            auto big = get_group(device, group, "biases_input"); load(device, g->biases_input, big, "parameters");
            auto whg = get_group(device, group, "weights_hidden"); load(device, g->weights_hidden, whg, "parameters");
            auto bhg = get_group(device, group, "biases_hidden"); load(device, g->biases_hidden, bhg, "parameters");
            auto ihg = get_group(device, group, "initial_hidden_state"); load(device, g->initial_hidden_state, ihg, "parameters");
            g->hidden_dim = g->weights_input.shape[0] / 3; g->input_dim = g->weights_input.shape[1];
            layer.data = g;
        }
        else if(utils::string::compare(type_str, "conv2d", 6)){
            layer.type = LayerType::CONV2D;
            auto* c = new layers::Conv2d<TI>();
            auto wg = get_group(device, group, "weights"); load(device, c->weights, wg, "parameters");
            auto bg = get_group(device, group, "biases"); load(device, c->biases, bg, "parameters");
            c->output_channels = get_attribute_int<TI>(device, group, "output_channels");
            c->input_channels = get_attribute_int<TI>(device, group, "input_channels");
            c->kernel_height = get_attribute_int<TI>(device, group, "kernel_height");
            c->kernel_width = get_attribute_int<TI>(device, group, "kernel_width");
            c->stride_h = get_attribute_int<TI>(device, group, "stride_h");
            c->stride_w = get_attribute_int<TI>(device, group, "stride_w");
            c->padding_h = get_attribute_int<TI>(device, group, "padding_h");
            c->padding_w = get_attribute_int<TI>(device, group, "padding_w");
            char af[ATTR_SIZE]; get_attribute<char*>(device, group, "activation_function", af, ATTR_SIZE);
            c->activation_function = persist_helpers::parse_activation_function(device, af, ATTR_SIZE);
            char norm[ATTR_SIZE]; get_attribute<char*>(device, group, "normalization", norm, ATTR_SIZE);
            if(utils::string::compare(norm, "BATCH_NORM", 10)) c->normalization = layers::Conv2d<TI>::Normalization::BATCH_NORM;
            else if(utils::string::compare(norm, "LAYER_NORM", 10)) c->normalization = layers::Conv2d<TI>::Normalization::LAYER_NORM;
            else c->normalization = layers::Conv2d<TI>::Normalization::NONE;
            if(c->normalization != layers::Conv2d<TI>::Normalization::NONE){
                auto gg = get_group(device, group, "gamma"); load(device, c->gamma, gg, "parameters");
                auto btg = get_group(device, group, "beta"); load(device, c->beta, btg, "parameters");
                if(c->normalization == layers::Conv2d<TI>::Normalization::BATCH_NORM){
                    load(device, c->running_mean, group, "running_mean");
                    load(device, c->running_var, group, "running_var");
                }
            }
            layer.data = c;
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
        }
        else if(utils::string::compare(type_str, "standardize", 11)){
            layer.type = LayerType::STANDARDIZE;
            auto* s = new layers::Standardize<TI>();
            auto mg = get_group(device, group, "mean"); load(device, s->mean, mg, "parameters");
            auto pg = get_group(device, group, "precision"); load(device, s->precision, pg, "parameters");
            s->dim = s->mean.size; layer.data = s;
        }
        else if(utils::string::compare(type_str, "embedding", 9)){
            layer.type = LayerType::EMBEDDING;
            auto* e = new layers::Embedding<TI>();
            auto wg = get_group(device, group, "weights"); load(device, e->weights, wg, "parameters");
            e->num_classes = e->weights.shape[0]; e->embedding_dim = e->weights.shape[1]; layer.data = e;
        }
        else if(utils::string::compare(type_str, "sample_and_squash", 17)){ layer.type = LayerType::SAMPLE_AND_SQUASH; }
        else if(utils::string::compare(type_str, "flatten", 7)){ layer.type = LayerType::FLATTEN; }
        else if(utils::string::compare(type_str, "unflatten", 9)){ layer.type = LayerType::UNFLATTEN; }
        else if(utils::string::compare(type_str, "avg_pool2d", 10)){ layer.type = LayerType::AVG_POOL2D; }
        // --- Composites: populate children ---
        else if(utils::string::compare(type_str, "sequential", 10)){
            layer.type = LayerType::SEQUENTIAL;
            auto layers_group = get_group(device, group, "layers");
            TI count = 0;
            for(TI i = 0; i < 100; i++){ char idx[10]; utils::string::int_to_string<long int, TI>(idx, 10, i); if(!group_exists(device, layers_group, idx)) break; count++; }
            layer.num_children = count; layer.children = new Layer<TI>[count];
            for(TI i = 0; i < count; i++){ char idx[10]; utils::string::int_to_string<long int, TI>(idx, 10, i); auto lg = get_group(device, layers_group, idx); load(device, layer.children[i], lg); }
        }
        else if(utils::string::compare(type_str, "mlp", 3)){
            layer.type = LayerType::MLP;
            TI num_layers = get_attribute_int<TI>(device, group, "num_layers");
            layer.num_children = num_layers; layer.children = new Layer<TI>[num_layers];
            auto ig = get_group(device, group, "input_layer"); load(device, layer.children[0], ig);
            auto hg = get_group(device, group, "hidden_layers");
            for(TI i = 0; i < num_layers - 2; i++){ char idx[10]; utils::string::int_to_string<long int, TI>(idx, 10, i); auto hlg = get_group(device, hg, idx); load(device, layer.children[1+i], hlg); }
            auto og = get_group(device, group, "output_layer"); load(device, layer.children[num_layers-1], og);
        }
        else if(utils::string::compare(type_str, "resnet_block", 12)){
            layer.type = LayerType::RESNET_BLOCK;
            bool has_ds = group_exists(device, group, "downsample");
            layer.num_children = has_ds ? 3 : 2; layer.children = new Layer<TI>[layer.num_children];
            auto c1g = get_group(device, group, "conv1"); load(device, layer.children[0], c1g);
            auto c2g = get_group(device, group, "conv2"); load(device, layer.children[1], c2g);
            if(has_ds){ auto dsg = get_group(device, group, "downsample"); load(device, layer.children[2], dsg); }
        }
        else if(utils::string::compare(type_str, "parallel", 8)){
            layer.type = LayerType::PARALLEL;
            bool has_head = group_exists(device, group, "head");
            layer.num_children = has_head ? 3 : 2; layer.children = new Layer<TI>[layer.num_children];
            auto pag = get_group(device, group, "pipeline_a"); load(device, layer.children[0], pag);
            auto pbg = get_group(device, group, "pipeline_b"); load(device, layer.children[1], pbg);
            if(has_head){ auto hg = get_group(device, group, "head"); load(device, layer.children[2], hg); }
        }
        else{ return false; }
        return true;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif

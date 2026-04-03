#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_DYN_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_DYN_OPERATIONS_GENERIC_H

#include "model.h"

#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, dyn::Tensor<SPEC>& tensor){
        using TI = typename SPEC::TI;
        TI total_bytes = tensor.size * dyn::size_of<TI>(tensor.type);
        if(total_bytes > 0){
            tensor.data = new char[total_bytes];
        }
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, dyn::Tensor<SPEC>& tensor){
        if(tensor.data != nullptr){
            delete[] reinterpret_cast<char*>(tensor.data);
            tensor.data = nullptr;
        }
    }

    namespace dyn{
        template <typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void set_shape(Tensor<TensorSpecification<TI>>& tensor, TI rank, const TI* shape){
            tensor.rank = rank;
            tensor.size = 1;
            for(TI i = 0; i < rank; i++){
                tensor.shape[i] = shape[i];
                tensor.size *= shape[i];
            }
        }
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT float get(DEVICE& device, const Tensor<TensorSpecification<TI>>& tensor, TI flat_index){
            TI element_size = size_of<TI>(tensor.type);
            const char* ptr = reinterpret_cast<const char*>(tensor.data) + flat_index * element_size;
            return to_float(ptr, tensor.type);
        }
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void set(DEVICE& device, Tensor<TensorSpecification<TI>>& tensor, TI flat_index, float value){
            TI element_size = size_of<TI>(tensor.type);
            char* ptr = reinterpret_cast<char*>(tensor.data) + flat_index * element_size;
            from_float(ptr, value, tensor.type);
        }

        RL_TOOLS_FUNCTION_PLACEMENT inline float apply_activation(ActivationFunction af, float x){
            switch(af){
                case ActivationFunction::IDENTITY: return x;
                case ActivationFunction::RELU: return x > 0 ? x : 0;
                case ActivationFunction::GELU: {
                    float c = 0.7978845608028654f;
                    float inner = c * (x + 0.044715f * x * x * x);
                    float t = inner > 10.0f ? 1.0f : (inner < -10.0f ? -1.0f : ((1.0f - 2.0f / (1.0f + __builtin_expf(2.0f * inner)))));
                    return 0.5f * x * (1.0f + t);
                }
                case ActivationFunction::TANH: {
                    if(x > 10.0f) return 1.0f;
                    if(x < -10.0f) return -1.0f;
                    float e2x = __builtin_expf(2.0f * x);
                    return (e2x - 1.0f) / (e2x + 1.0f);
                }
                case ActivationFunction::FAST_TANH: {
                    float clamped = x < -3.0f ? -3.0f : (x > 3.0f ? 3.0f : x);
                    float x_squared = clamped * clamped;
                    return clamped * (27.0f + x_squared) / (27.0f + 9.0f * x_squared);
                }
                case ActivationFunction::SIGMOID: return 1.0f / (1.0f + __builtin_expf(-x));
                default: return x;
            }
        }

        // --- Shape propagation (one-time init) ---
        template <typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void propagate_shapes(Layer<TI>& layer, const TI* in_shape, TI in_rank, TI in_size){
            switch(layer.type){
                case LayerType::DENSE: {
                    auto* d = reinterpret_cast<const layers::Dense<TI>*>(layer.data);
                    layer.output_rank = in_rank;
                    layer.output_size = 1;
                    for(TI i = 0; i < in_rank - 1; i++){ layer.output_shape[i] = in_shape[i]; layer.output_size *= in_shape[i]; }
                    layer.output_shape[in_rank - 1] = d->output_dim;
                    layer.output_size *= d->output_dim;
                    break;
                }
                case LayerType::GRU: {
                    auto* g = reinterpret_cast<const layers::GRU<TI>*>(layer.data);
                    layer.output_rank = in_rank;
                    layer.output_size = 1;
                    for(TI i = 0; i < in_rank - 1; i++){ layer.output_shape[i] = in_shape[i]; layer.output_size *= in_shape[i]; }
                    layer.output_shape[in_rank - 1] = g->hidden_dim;
                    layer.output_size *= g->hidden_dim;
                    break;
                }
                case LayerType::CONV2D: {
                    auto* c = reinterpret_cast<const layers::Conv2d<TI>*>(layer.data);
                    TI ih = in_shape[in_rank - 3], iw = in_shape[in_rank - 2];
                    TI oh = (ih + 2 * c->padding_h - c->kernel_height) / c->stride_h + 1;
                    TI ow = (iw + 2 * c->padding_w - c->kernel_width) / c->stride_w + 1;
                    TI batch = in_size / (ih * iw * c->input_channels);
                    layer.output_rank = 4;
                    layer.output_shape[0] = batch; layer.output_shape[1] = oh; layer.output_shape[2] = ow; layer.output_shape[3] = c->output_channels;
                    layer.output_size = batch * oh * ow * c->output_channels;
                    break;
                }
                case LayerType::MAX_POOL2D: {
                    auto* mp = reinterpret_cast<const layers::MaxPool2d<TI>*>(layer.data);
                    TI ih = in_shape[in_rank - 3], iw = in_shape[in_rank - 2], ch = in_shape[in_rank - 1];
                    TI oh = (ih + 2 * mp->padding_h - mp->kernel_height) / mp->stride_h + 1;
                    TI ow = (iw + 2 * mp->padding_w - mp->kernel_width) / mp->stride_w + 1;
                    TI batch = in_size / (ih * iw * ch);
                    layer.output_rank = 4;
                    layer.output_shape[0] = batch; layer.output_shape[1] = oh; layer.output_shape[2] = ow; layer.output_shape[3] = ch;
                    layer.output_size = batch * oh * ow * ch;
                    break;
                }
                case LayerType::AVG_POOL2D: {
                    TI ch = in_shape[in_rank - 1], ih = in_shape[in_rank - 3], iw = in_shape[in_rank - 2];
                    TI batch = in_size / (ih * iw * ch);
                    layer.output_rank = 2; layer.output_shape[0] = batch; layer.output_shape[1] = ch;
                    layer.output_size = batch * ch;
                    break;
                }
                case LayerType::FLATTEN: {
                    if(in_rank >= 3){
                        TI flat = 1; for(TI i = in_rank - 3; i < in_rank; i++) flat *= in_shape[i];
                        TI batch = in_size / flat;
                        layer.output_rank = 2; layer.output_shape[0] = batch; layer.output_shape[1] = flat;
                    } else {
                        layer.output_rank = in_rank; for(TI i = 0; i < in_rank; i++) layer.output_shape[i] = in_shape[i];
                    }
                    layer.output_size = in_size;
                    break;
                }
                case LayerType::SAMPLE_AND_SQUASH: {
                    TI last = in_shape[in_rank - 1], batch = in_size / last;
                    layer.output_rank = 2; layer.output_shape[0] = batch; layer.output_shape[1] = last / 2;
                    layer.output_size = batch * (last / 2);
                    break;
                }
                case LayerType::STANDARDIZE: {
                    layer.output_rank = in_rank; layer.output_size = in_size;
                    for(TI i = 0; i < in_rank; i++) layer.output_shape[i] = in_shape[i];
                    break;
                }
                case LayerType::SEQUENTIAL: {
                    auto* seq = reinterpret_cast<layers::Sequential<TI>*>(layer.data);
                    const TI* cur_shape = in_shape; TI cur_rank = in_rank, cur_size = in_size;
                    for(TI i = 0; i < seq->num_layers; i++){
                        propagate_shapes(seq->layers[i], cur_shape, cur_rank, cur_size);
                        cur_shape = seq->layers[i].output_shape;
                        cur_rank = seq->layers[i].output_rank;
                        cur_size = seq->layers[i].output_size;
                    }
                    layer.output_rank = cur_rank; layer.output_size = cur_size;
                    for(TI i = 0; i < cur_rank; i++) layer.output_shape[i] = cur_shape[i];
                    break;
                }
                case LayerType::MLP: {
                    auto* mlp = reinterpret_cast<layers::MLP<TI>*>(layer.data);
                    propagate_shapes(mlp->input_layer, in_shape, in_rank, in_size);
                    const TI* cur_shape = mlp->input_layer.output_shape;
                    TI cur_rank = mlp->input_layer.output_rank, cur_size = mlp->input_layer.output_size;
                    for(TI i = 0; i < mlp->num_hidden_layers; i++){
                        propagate_shapes(mlp->hidden_layers[i], cur_shape, cur_rank, cur_size);
                        cur_shape = mlp->hidden_layers[i].output_shape;
                        cur_rank = mlp->hidden_layers[i].output_rank;
                        cur_size = mlp->hidden_layers[i].output_size;
                    }
                    propagate_shapes(mlp->output_layer, cur_shape, cur_rank, cur_size);
                    layer.output_rank = mlp->output_layer.output_rank;
                    layer.output_size = mlp->output_layer.output_size;
                    for(TI i = 0; i < layer.output_rank; i++) layer.output_shape[i] = mlp->output_layer.output_shape[i];
                    break;
                }
                case LayerType::RESNET_BLOCK: {
                    auto* rb = reinterpret_cast<layers::ResnetBlock<TI>*>(layer.data);
                    propagate_shapes(rb->conv1, in_shape, in_rank, in_size);
                    propagate_shapes(rb->conv2, rb->conv1.output_shape, rb->conv1.output_rank, rb->conv1.output_size);
                    if(rb->downsample) propagate_shapes(*rb->downsample, in_shape, in_rank, in_size);
                    layer.output_rank = rb->conv2.output_rank;
                    layer.output_size = rb->conv2.output_size;
                    for(TI i = 0; i < layer.output_rank; i++) layer.output_shape[i] = rb->conv2.output_shape[i];
                    break;
                }
                default: {
                    layer.output_rank = in_rank; layer.output_size = in_size;
                    for(TI i = 0; i < in_rank; i++) layer.output_shape[i] = in_shape[i];
                    break;
                }
            }
        }

        // --- Buffer sizing helpers ---
        template <typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT TI max_output_size(const Layer<TI>& layer){
            TI m = layer.output_size;
            switch(layer.type){
                case LayerType::SEQUENTIAL: {
                    auto* seq = reinterpret_cast<const layers::Sequential<TI>*>(layer.data);
                    for(TI i = 0; i < seq->num_layers; i++){ TI s = max_output_size(seq->layers[i]); if(s > m) m = s; }
                    break;
                }
                case LayerType::MLP: {
                    auto* mlp = reinterpret_cast<const layers::MLP<TI>*>(layer.data);
                    TI s = max_output_size(mlp->input_layer); if(s > m) m = s;
                    for(TI i = 0; i < mlp->num_hidden_layers; i++){ s = max_output_size(mlp->hidden_layers[i]); if(s > m) m = s; }
                    s = max_output_size(mlp->output_layer); if(s > m) m = s;
                    break;
                }
                case LayerType::RESNET_BLOCK: {
                    auto* rb = reinterpret_cast<const layers::ResnetBlock<TI>*>(layer.data);
                    TI s = max_output_size(rb->conv1); if(s > m) m = s;
                    s = max_output_size(rb->conv2); if(s > m) m = s;
                    if(rb->downsample){ s = max_output_size(*rb->downsample); if(s > m) m = s; }
                    break;
                }
                default: break;
            }
            return m;
        }

        template <typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT TI max_scratch_size(const Layer<TI>& layer){
            TI m = 0;
            switch(layer.type){
                case LayerType::GRU: {
                    auto* g = reinterpret_cast<const layers::GRU<TI>*>(layer.data);
                    TI batch = (layer.output_rank == 3) ? layer.output_shape[1] : (layer.output_size / g->hidden_dim);
                    m = batch * 3 * g->hidden_dim;
                    break;
                }
                case LayerType::RESNET_BLOCK: {
                    auto* rb = reinterpret_cast<const layers::ResnetBlock<TI>*>(layer.data);
                    m = rb->conv1.output_size;
                    break;
                }
                case LayerType::SEQUENTIAL: {
                    auto* seq = reinterpret_cast<const layers::Sequential<TI>*>(layer.data);
                    for(TI i = 0; i < seq->num_layers; i++){ TI s = max_scratch_size(seq->layers[i]); if(s > m) m = s; }
                    break;
                }
                case LayerType::MLP: {
                    auto* mlp = reinterpret_cast<const layers::MLP<TI>*>(layer.data);
                    TI s = max_scratch_size(mlp->input_layer); if(s > m) m = s;
                    for(TI i = 0; i < mlp->num_hidden_layers; i++){ s = max_scratch_size(mlp->hidden_layers[i]); if(s > m) m = s; }
                    s = max_scratch_size(mlp->output_layer); if(s > m) m = s;
                    break;
                }
                default: break;
            }
            return m;
        }

        // --- Layer evaluate helpers ---
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_dense(DEVICE& device, const layers::Dense<TI>& layer, const Tensor<TensorSpecification<TI>>& input, Tensor<TensorSpecification<TI>>& output){
            TI batch_size = input.size / layer.input_dim;
            for(TI b = 0; b < batch_size; b++){
                for(TI o = 0; o < layer.output_dim; o++){
                    float acc = get(device, layer.biases, o);
                    for(TI i = 0; i < layer.input_dim; i++){
                        acc += get(device, layer.weights, o * layer.input_dim + i) * get(device, input, b * layer.input_dim + i);
                    }
                    set(device, output, b * layer.output_dim + o, apply_activation(layer.activation_function, acc));
                }
            }
        }
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step_gru(DEVICE& device, const layers::GRU<TI>& layer, const Tensor<TensorSpecification<TI>>& input, state::GRU<TI>& gru_state, Tensor<TensorSpecification<TI>>& scratch){
            TI batch_size = input.size / layer.input_dim;
            TI hidden_dim = layer.hidden_dim;
            if(!gru_state.initialized){
                for(TI b = 0; b < batch_size; b++)
                    for(TI h = 0; h < hidden_dim; h++)
                        set(device, gru_state.hidden, b * hidden_dim + h, get(device, layer.initial_hidden_state, h));
                gru_state.initialized = true;
            }
            for(TI b = 0; b < batch_size; b++){
                for(TI g = 0; g < 2 * hidden_dim; g++){
                    float wh = get(device, layer.biases_hidden, g);
                    for(TI h = 0; h < hidden_dim; h++) wh += get(device, layer.weights_hidden, g * hidden_dim + h) * get(device, gru_state.hidden, b * hidden_dim + h);
                    float wi = get(device, layer.biases_input, g);
                    for(TI i = 0; i < layer.input_dim; i++) wi += get(device, layer.weights_input, g * layer.input_dim + i) * get(device, input, b * layer.input_dim + i);
                    set(device, scratch, b * 2 * hidden_dim + g, 1.0f / (1.0f + __builtin_expf(-(wh + wi))));
                }
                for(TI h = 0; h < hidden_dim; h++){
                    TI g = 2 * hidden_dim + h;
                    float wh_n = get(device, layer.biases_hidden, g);
                    for(TI hh = 0; hh < hidden_dim; hh++) wh_n += get(device, layer.weights_hidden, g * hidden_dim + hh) * get(device, gru_state.hidden, b * hidden_dim + hh);
                    float wi_n = get(device, layer.biases_input, g);
                    for(TI i = 0; i < layer.input_dim; i++) wi_n += get(device, layer.weights_input, g * layer.input_dim + i) * get(device, input, b * layer.input_dim + i);
                    float r = get(device, scratch, b * 2 * hidden_dim + h);
                    float n_pre = r * wh_n + wi_n;
                    float n;
                    if(n_pre > 10.0f) n = 1.0f; else if(n_pre < -10.0f) n = -1.0f;
                    else{ float e2x = __builtin_expf(2.0f * n_pre); n = (e2x - 1.0f) / (e2x + 1.0f); }
                    float z = get(device, scratch, b * 2 * hidden_dim + hidden_dim + h);
                    float new_h = z * get(device, gru_state.hidden, b * hidden_dim + h) + (1.0f - z) * n;
                    set(device, scratch, b * 2 * hidden_dim + h, new_h);
                }
                for(TI h = 0; h < hidden_dim; h++)
                    set(device, gru_state.hidden, b * hidden_dim + h, get(device, scratch, b * 2 * hidden_dim + h));
            }
        }
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_conv2d(DEVICE& device, const layers::Conv2d<TI>& layer, const Tensor<TensorSpecification<TI>>& input, Tensor<TensorSpecification<TI>>& output){
            TI input_height = input.shape[input.rank - 3], input_width = input.shape[input.rank - 2];
            TI batch_size = input.size / (input_height * input_width * layer.input_channels);
            TI output_height = (input_height + 2 * layer.padding_h - layer.kernel_height) / layer.stride_h + 1;
            TI output_width = (input_width + 2 * layer.padding_w - layer.kernel_width) / layer.stride_w + 1;
            for(TI bi = 0; bi < batch_size; bi++)
                for(TI oh = 0; oh < output_height; oh++)
                    for(TI ow = 0; ow < output_width; ow++)
                        for(TI oc = 0; oc < layer.output_channels; oc++){
                            float acc = get(device, layer.biases, oc);
                            for(TI kh = 0; kh < layer.kernel_height; kh++){
                                TI ih = oh * layer.stride_h + kh - layer.padding_h;
                                if(ih >= input_height) continue;
                                for(TI kw = 0; kw < layer.kernel_width; kw++){
                                    TI iw = ow * layer.stride_w + kw - layer.padding_w;
                                    if(iw >= input_width) continue;
                                    for(TI ic = 0; ic < layer.input_channels; ic++)
                                        acc += get(device, layer.weights, ((oc * layer.kernel_height + kh) * layer.kernel_width + kw) * layer.input_channels + ic)
                                             * get(device, input, ((bi * input_height + ih) * input_width + iw) * layer.input_channels + ic);
                                }
                            }
                            set(device, output, ((bi * output_height + oh) * output_width + ow) * layer.output_channels + oc, acc);
                        }
            if(layer.normalization == layers::Conv2d<TI>::Normalization::BATCH_NORM){
                float epsilon = 1e-5f;
                for(TI i = 0; i < output.size; i++){
                    TI oc = i % layer.output_channels;
                    float x = get(device, output, i);
                    set(device, output, i, get(device, layer.gamma, oc) * (x - get(device, layer.running_mean, oc)) / __builtin_sqrtf(get(device, layer.running_var, oc) + epsilon) + get(device, layer.beta, oc));
                }
            }
            else if(layer.normalization == layers::Conv2d<TI>::Normalization::LAYER_NORM){
                float epsilon = 1e-5f;
                TI spatial_channels = output.size / batch_size;
                for(TI bi = 0; bi < batch_size; bi++){
                    float sum = 0;
                    for(TI i = 0; i < spatial_channels; i++) sum += get(device, output, bi * spatial_channels + i);
                    float mean = sum / (float)spatial_channels;
                    float var_sum = 0;
                    for(TI i = 0; i < spatial_channels; i++){ float d = get(device, output, bi * spatial_channels + i) - mean; var_sum += d * d; }
                    float inv_std = 1.0f / __builtin_sqrtf(var_sum / (float)spatial_channels + epsilon);
                    for(TI i = 0; i < spatial_channels; i++){
                        TI oc = i % layer.output_channels;
                        set(device, output, bi * spatial_channels + i, get(device, layer.gamma, oc) * (get(device, output, bi * spatial_channels + i) - mean) * inv_std + get(device, layer.beta, oc));
                    }
                }
            }
            if(layer.activation_function != ActivationFunction::IDENTITY)
                for(TI i = 0; i < output.size; i++) set(device, output, i, apply_activation(layer.activation_function, get(device, output, i)));
        }
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_max_pool2d(DEVICE& device, const layers::MaxPool2d<TI>& layer, const Tensor<TensorSpecification<TI>>& input, Tensor<TensorSpecification<TI>>& output){
            TI input_height = input.shape[input.rank - 3], input_width = input.shape[input.rank - 2], channels = input.shape[input.rank - 1];
            TI batch_size = input.size / (input_height * input_width * channels);
            TI output_height = (input_height + 2 * layer.padding_h - layer.kernel_height) / layer.stride_h + 1;
            TI output_width = (input_width + 2 * layer.padding_w - layer.kernel_width) / layer.stride_w + 1;
            for(TI bi = 0; bi < batch_size; bi++)
                for(TI oh = 0; oh < output_height; oh++)
                    for(TI ow = 0; ow < output_width; ow++)
                        for(TI c = 0; c < channels; c++){
                            float max_val = -1e30f;
                            for(TI kh = 0; kh < layer.kernel_height; kh++){
                                TI ih = oh * layer.stride_h + kh - layer.padding_h;
                                if(ih >= input_height) continue;
                                for(TI kw = 0; kw < layer.kernel_width; kw++){
                                    TI iw = ow * layer.stride_w + kw - layer.padding_w;
                                    if(iw >= input_width) continue;
                                    float val = get(device, input, ((bi * input_height + ih) * input_width + iw) * channels + c);
                                    if(val > max_val) max_val = val;
                                }
                            }
                            set(device, output, ((bi * output_height + oh) * output_width + ow) * channels + c, max_val);
                        }
        }
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_sample_and_squash(DEVICE& device, const Tensor<TensorSpecification<TI>>& input, Tensor<TensorSpecification<TI>>& output){
            TI last_dim = input.shape[input.rank - 1], half_dim = last_dim / 2, batch_size = input.size / last_dim;
            for(TI b = 0; b < batch_size; b++)
                for(TI i = 0; i < half_dim; i++){
                    float mean = get(device, input, b * last_dim + i);
                    float t; if(mean > 10.0f) t = 1.0f; else if(mean < -10.0f) t = -1.0f;
                    else{ float e2x = __builtin_expf(2.0f * mean); t = (e2x - 1.0f) / (e2x + 1.0f); }
                    set(device, output, b * half_dim + i, t);
                }
        }
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_standardize(DEVICE& device, const layers::Standardize<TI>& layer, const Tensor<TensorSpecification<TI>>& input, Tensor<TensorSpecification<TI>>& output){
            TI batch_size = input.size / layer.dim;
            for(TI b = 0; b < batch_size; b++)
                for(TI i = 0; i < layer.dim; i++)
                    set(device, output, b * layer.dim + i, (get(device, input, b * layer.dim + i) - get(device, layer.mean, i)) * get(device, layer.precision, i));
        }
    }

    // Forward declarations
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, const dyn::Layer<TI>& layer, const dyn::Tensor<dyn::TensorSpecification<TI>>& input, dyn::Tensor<dyn::TensorSpecification<TI>>& output, dyn::Buffer<TI>& buffer);
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step(DEVICE& device, const dyn::Layer<TI>& layer, const dyn::Tensor<dyn::TensorSpecification<TI>>& input, dyn::State<TI>& state, dyn::Tensor<dyn::TensorSpecification<TI>>& output, dyn::Buffer<TI>& buffer);

    namespace dyn{
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_sequential(DEVICE& device, const layers::Sequential<TI>& seq, const Tensor<TensorSpecification<TI>>& input, Tensor<TensorSpecification<TI>>& output, Buffer<TI>& buffer){
            const Tensor<TensorSpecification<TI>>* current_input = &input;
            bool input_is_external = true;
            for(TI i = 0; i < seq.num_layers; i++){
                Tensor<TensorSpecification<TI>>* current_output;
                if(i == seq.num_layers - 1) current_output = &output;
                else current_output = input_is_external ? &buffer.tick : (current_input == &buffer.tick ? &buffer.tock : &buffer.tick);
                rl_tools::evaluate(device, seq.layers[i], *current_input, *current_output, buffer);
                current_input = current_output;
                input_is_external = false;
            }
        }
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step_sequential(DEVICE& device, const layers::Sequential<TI>& seq, state::Sequential<TI>& seq_state, const Tensor<TensorSpecification<TI>>& input, Tensor<TensorSpecification<TI>>& output, Buffer<TI>& buffer){
            const Tensor<TensorSpecification<TI>>* current_input = &input;
            bool input_is_external = true;
            for(TI i = 0; i < seq.num_layers; i++){
                Tensor<TensorSpecification<TI>>* current_output;
                if(i == seq.num_layers - 1) current_output = &output;
                else current_output = input_is_external ? &buffer.tick : (current_input == &buffer.tick ? &buffer.tock : &buffer.tick);
                rl_tools::evaluate_step(device, seq.layers[i], *current_input, seq_state.layer_states[i], *current_output, buffer);
                current_input = current_output;
                input_is_external = false;
            }
        }
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_mlp(DEVICE& device, const layers::MLP<TI>& mlp, const Tensor<TensorSpecification<TI>>& input, Tensor<TensorSpecification<TI>>& output, Buffer<TI>& buffer){
            TI total_layers = 1 + mlp.num_hidden_layers + 1;
            const Tensor<TensorSpecification<TI>>* current_input = &input;
            bool input_is_external = true;
            for(TI i = 0; i < total_layers; i++){
                const Layer<TI>* current_layer;
                if(i == 0) current_layer = &mlp.input_layer;
                else if(i <= mlp.num_hidden_layers) current_layer = &mlp.hidden_layers[i - 1];
                else current_layer = &mlp.output_layer;
                Tensor<TensorSpecification<TI>>* current_output;
                if(i == total_layers - 1) current_output = &output;
                else current_output = input_is_external ? &buffer.tick : (current_input == &buffer.tick ? &buffer.tock : &buffer.tick);
                rl_tools::evaluate(device, *current_layer, *current_input, *current_output, buffer);
                current_input = current_output;
                input_is_external = false;
            }
        }
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_gru(DEVICE& device, const layers::GRU<TI>& layer, const Layer<TI>& layer_meta, const Tensor<TensorSpecification<TI>>& input, Tensor<TensorSpecification<TI>>& output, Buffer<TI>& buffer){
            TI hidden_dim = layer.hidden_dim;
            TI batch = (input.rank == 3) ? input.shape[1] : (input.size / layer.input_dim);
            // Pack hidden state and gate scratch into buffer.scratch
            // scratch[0..batch*H-1] = hidden state, scratch[batch*H..batch*3H-1] = gates
            state::GRU<TI> tmp_state;
            TI state_shape[] = {batch, hidden_dim};
            set_shape(tmp_state.hidden, (TI)2, state_shape);
            tmp_state.hidden.type = Type::FLOAT32;
            tmp_state.hidden.data = buffer.scratch.data;
            tmp_state.initialized = false;
            Tensor<TensorSpecification<TI>> gate_scratch;
            TI gate_shape[] = {batch * 2 * hidden_dim};
            set_shape(gate_scratch, (TI)1, gate_shape);
            gate_scratch.type = Type::FLOAT32;
            gate_scratch.data = reinterpret_cast<char*>(buffer.scratch.data) + batch * hidden_dim * sizeof(float);
            if(input.rank == 2){
                evaluate_step_gru(device, layer, input, tmp_state, gate_scratch);
                for(TI b = 0; b < batch; b++)
                    for(TI h = 0; h < hidden_dim; h++)
                        set(device, output, b * hidden_dim + h, get(device, tmp_state.hidden, b * hidden_dim + h));
            }
            else{
                TI seq_len = input.shape[0];
                Tensor<TensorSpecification<TI>> step_input;
                TI step_shape[] = {batch, layer.input_dim};
                set_shape(step_input, (TI)2, step_shape);
                step_input.type = input.type;
                for(TI t = 0; t < seq_len; t++){
                    step_input.data = reinterpret_cast<char*>(const_cast<void*>(input.data)) + t * batch * layer.input_dim * size_of<TI>(input.type);
                    evaluate_step_gru(device, layer, step_input, tmp_state, gate_scratch);
                    for(TI b = 0; b < batch; b++)
                        for(TI h = 0; h < hidden_dim; h++)
                            set(device, output, (t * batch + b) * hidden_dim + h, get(device, tmp_state.hidden, b * hidden_dim + h));
                }
            }
        }
    }

    // --- Buffer malloc/free ---
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, dyn::Buffer<TI>& buffer){
        TI tick_tock_size = dyn::max_output_size(*buffer.layer);
        TI scratch_size = dyn::max_scratch_size(*buffer.layer);
        TI tt_shape[] = {tick_tock_size};
        dyn::set_shape(buffer.tick, (TI)1, tt_shape);
        buffer.tick.type = dyn::Type::FLOAT32;
        rl_tools::malloc(device, buffer.tick);
        dyn::set_shape(buffer.tock, (TI)1, tt_shape);
        buffer.tock.type = dyn::Type::FLOAT32;
        rl_tools::malloc(device, buffer.tock);
        if(scratch_size > 0){
            TI s_shape[] = {scratch_size};
            dyn::set_shape(buffer.scratch, (TI)1, s_shape);
            buffer.scratch.type = dyn::Type::FLOAT32;
            rl_tools::malloc(device, buffer.scratch);
        }
    }
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, dyn::Buffer<TI>& buffer){
        rl_tools::free(device, buffer.tick);
        rl_tools::free(device, buffer.tock);
        rl_tools::free(device, buffer.scratch);
    }

    // --- State malloc/free/reset ---
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, dyn::State<TI>& state){
        state.type = state.layer->type;
        switch(state.type){
            case dyn::LayerType::GRU: {
                auto* gru_layer = reinterpret_cast<const dyn::layers::GRU<TI>*>(state.layer->data);
                auto* gs = new dyn::state::GRU<TI>();
                TI shape[] = {state.batch_size, gru_layer->hidden_dim};
                dyn::set_shape(gs->hidden, (TI)2, shape);
                gs->hidden.type = dyn::Type::FLOAT32;
                rl_tools::malloc(device, gs->hidden);
                gs->initialized = false;
                state.data = gs;
                break;
            }
            case dyn::LayerType::SEQUENTIAL: {
                auto* seq = reinterpret_cast<const dyn::layers::Sequential<TI>*>(state.layer->data);
                auto* ss = new dyn::state::Sequential<TI>();
                ss->num_layers = seq->num_layers;
                ss->layer_states = new dyn::State<TI>[seq->num_layers];
                for(TI i = 0; i < seq->num_layers; i++){
                    ss->layer_states[i].batch_size = state.batch_size;
                    ss->layer_states[i].layer = &seq->layers[i];
                    rl_tools::malloc(device, ss->layer_states[i]);
                }
                state.data = ss;
                break;
            }
            default: state.data = nullptr; break;
        }
    }
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, dyn::State<TI>& state){
        switch(state.type){
            case dyn::LayerType::GRU: {
                auto* gs = reinterpret_cast<dyn::state::GRU<TI>*>(state.data);
                rl_tools::free(device, gs->hidden);
                delete gs;
                break;
            }
            case dyn::LayerType::SEQUENTIAL: {
                auto* ss = reinterpret_cast<dyn::state::Sequential<TI>*>(state.data);
                for(TI i = 0; i < ss->num_layers; i++) rl_tools::free(device, ss->layer_states[i]);
                delete[] ss->layer_states;
                delete ss;
                break;
            }
            default: break;
        }
        state.data = nullptr;
    }
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE& device, const dyn::Layer<TI>& layer, dyn::State<TI>& state){
        switch(state.type){
            case dyn::LayerType::GRU: reinterpret_cast<dyn::state::GRU<TI>*>(state.data)->initialized = false; break;
            case dyn::LayerType::SEQUENTIAL: {
                auto* ss = reinterpret_cast<dyn::state::Sequential<TI>*>(state.data);
                auto* seq = reinterpret_cast<const dyn::layers::Sequential<TI>*>(layer.data);
                for(TI i = 0; i < ss->num_layers; i++) reset(device, seq->layers[i], ss->layer_states[i]);
                break;
            }
            default: break;
        }
    }

    // --- Layer free ---
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, dyn::Layer<TI>& layer){
        switch(layer.type){
            case dyn::LayerType::DENSE: { auto* d = reinterpret_cast<dyn::layers::Dense<TI>*>(layer.data); rl_tools::free(device, d->weights); rl_tools::free(device, d->biases); delete d; break; }
            case dyn::LayerType::GRU: { auto* g = reinterpret_cast<dyn::layers::GRU<TI>*>(layer.data); rl_tools::free(device, g->weights_input); rl_tools::free(device, g->biases_input); rl_tools::free(device, g->weights_hidden); rl_tools::free(device, g->biases_hidden); rl_tools::free(device, g->initial_hidden_state); delete g; break; }
            case dyn::LayerType::CONV2D: { auto* c = reinterpret_cast<dyn::layers::Conv2d<TI>*>(layer.data); rl_tools::free(device, c->weights); rl_tools::free(device, c->biases); if(c->normalization != dyn::layers::Conv2d<TI>::Normalization::NONE){ rl_tools::free(device, c->gamma); rl_tools::free(device, c->beta); if(c->normalization == dyn::layers::Conv2d<TI>::Normalization::BATCH_NORM){ rl_tools::free(device, c->running_mean); rl_tools::free(device, c->running_var); } } delete c; break; }
            case dyn::LayerType::MAX_POOL2D: delete reinterpret_cast<dyn::layers::MaxPool2d<TI>*>(layer.data); break;
            case dyn::LayerType::STANDARDIZE: { auto* s = reinterpret_cast<dyn::layers::Standardize<TI>*>(layer.data); rl_tools::free(device, s->mean); rl_tools::free(device, s->precision); delete s; break; }
            case dyn::LayerType::EMBEDDING: { auto* e = reinterpret_cast<dyn::layers::Embedding<TI>*>(layer.data); rl_tools::free(device, e->weights); delete e; break; }
            case dyn::LayerType::SEQUENTIAL: { auto* seq = reinterpret_cast<dyn::layers::Sequential<TI>*>(layer.data); for(TI i = 0; i < seq->num_layers; i++) rl_tools::free(device, seq->layers[i]); delete[] seq->layers; delete seq; break; }
            case dyn::LayerType::MLP: { auto* mlp = reinterpret_cast<dyn::layers::MLP<TI>*>(layer.data); rl_tools::free(device, mlp->input_layer); for(TI i = 0; i < mlp->num_hidden_layers; i++) rl_tools::free(device, mlp->hidden_layers[i]); delete[] mlp->hidden_layers; rl_tools::free(device, mlp->output_layer); delete mlp; break; }
            case dyn::LayerType::PARALLEL: { auto* p = reinterpret_cast<dyn::layers::Parallel<TI>*>(layer.data); rl_tools::free(device, *p->pipeline_a); delete p->pipeline_a; rl_tools::free(device, *p->pipeline_b); delete p->pipeline_b; if(p->head){ rl_tools::free(device, *p->head); delete p->head; } delete p; break; }
            case dyn::LayerType::RESNET_BLOCK: { auto* rb = reinterpret_cast<dyn::layers::ResnetBlock<TI>*>(layer.data); rl_tools::free(device, rb->conv1); rl_tools::free(device, rb->conv2); if(rb->downsample){ rl_tools::free(device, *rb->downsample); delete rb->downsample; } delete rb; break; }
            default: break;
        }
        layer.data = nullptr;
    }

    // --- Top-level evaluate ---
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, const dyn::Layer<TI>& layer, const dyn::Tensor<dyn::TensorSpecification<TI>>& input, dyn::Tensor<dyn::TensorSpecification<TI>>& output, dyn::Buffer<TI>& buffer){
        // Set output shape from cached propagation
        dyn::set_shape(output, layer.output_rank, layer.output_shape);
        output.type = dyn::Type::FLOAT32;
        switch(layer.type){
            case dyn::LayerType::DENSE: dyn::evaluate_dense(device, *reinterpret_cast<const dyn::layers::Dense<TI>*>(layer.data), input, output); break;
            case dyn::LayerType::GRU: dyn::evaluate_gru(device, *reinterpret_cast<const dyn::layers::GRU<TI>*>(layer.data), layer, input, output, buffer); break;
            case dyn::LayerType::CONV2D: dyn::evaluate_conv2d(device, *reinterpret_cast<const dyn::layers::Conv2d<TI>*>(layer.data), input, output); break;
            case dyn::LayerType::MAX_POOL2D: dyn::evaluate_max_pool2d(device, *reinterpret_cast<const dyn::layers::MaxPool2d<TI>*>(layer.data), input, output); break;
            case dyn::LayerType::SAMPLE_AND_SQUASH: dyn::evaluate_sample_and_squash(device, input, output); break;
            case dyn::LayerType::STANDARDIZE: dyn::evaluate_standardize(device, *reinterpret_cast<const dyn::layers::Standardize<TI>*>(layer.data), input, output); break;
            case dyn::LayerType::SEQUENTIAL: dyn::evaluate_sequential(device, *reinterpret_cast<const dyn::layers::Sequential<TI>*>(layer.data), input, output, buffer); break;
            case dyn::LayerType::MLP: dyn::evaluate_mlp(device, *reinterpret_cast<const dyn::layers::MLP<TI>*>(layer.data), input, output, buffer); break;
            case dyn::LayerType::FLATTEN:
            case dyn::LayerType::UNFLATTEN: {
                output.type = input.type;
                TI element_size = dyn::size_of<TI>(input.type);
                const char* src = reinterpret_cast<const char*>(input.data);
                char* dst = reinterpret_cast<char*>(output.data);
                for(TI i = 0; i < input.size * element_size; i++) dst[i] = src[i];
                break;
            }
            case dyn::LayerType::AVG_POOL2D: {
                TI channels = input.shape[input.rank - 1], width = input.shape[input.rank - 2], height = input.shape[input.rank - 3];
                TI batch_size = input.size / (height * width * channels);
                float scale = 1.0f / (float)(height * width);
                for(TI b = 0; b < batch_size; b++)
                    for(TI c = 0; c < channels; c++){
                        float sum = 0;
                        for(TI h = 0; h < height; h++)
                            for(TI w = 0; w < width; w++)
                                sum += dyn::get(device, input, ((b * height + h) * width + w) * channels + c);
                        dyn::set(device, output, b * channels + c, sum * scale);
                    }
                break;
            }
            case dyn::LayerType::RESNET_BLOCK: {
                auto* rb = reinterpret_cast<const dyn::layers::ResnetBlock<TI>*>(layer.data);
                rl_tools::evaluate(device, rb->conv1, input, buffer.scratch, buffer);
                rl_tools::evaluate(device, rb->conv2, buffer.scratch, output, buffer);
                if(rb->downsample){
                    rl_tools::evaluate(device, *rb->downsample, input, buffer.scratch, buffer);
                }
                const dyn::Tensor<dyn::TensorSpecification<TI>>& shortcut = rb->downsample ? buffer.scratch : input;
                for(TI i = 0; i < output.size; i++){
                    float val = dyn::get(device, output, i) + dyn::get(device, shortcut, i);
                    dyn::set(device, output, i, val > 0 ? val : 0);
                }
                break;
            }
            default: break;
        }
    }

    // --- Top-level evaluate_step ---
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step(DEVICE& device, const dyn::Layer<TI>& layer, const dyn::Tensor<dyn::TensorSpecification<TI>>& input, dyn::State<TI>& state, dyn::Tensor<dyn::TensorSpecification<TI>>& output, dyn::Buffer<TI>& buffer){
        switch(layer.type){
            case dyn::LayerType::GRU: {
                auto* g = reinterpret_cast<const dyn::layers::GRU<TI>*>(layer.data);
                auto* gru_state = reinterpret_cast<dyn::state::GRU<TI>*>(state.data);
                dyn::set_shape(output, layer.output_rank, layer.output_shape);
                output.type = dyn::Type::FLOAT32;
                dyn::Tensor<dyn::TensorSpecification<TI>> gate_scratch;
                TI batch_size = input.size / g->input_dim;
                TI gs[] = {batch_size * 2 * g->hidden_dim};
                dyn::set_shape(gate_scratch, (TI)1, gs);
                gate_scratch.type = dyn::Type::FLOAT32;
                gate_scratch.data = buffer.scratch.data;
                dyn::evaluate_step_gru(device, *g, input, *gru_state, gate_scratch);
                for(TI b = 0; b < batch_size; b++)
                    for(TI h = 0; h < g->hidden_dim; h++)
                        dyn::set(device, output, b * g->hidden_dim + h, dyn::get(device, gru_state->hidden, b * g->hidden_dim + h));
                break;
            }
            case dyn::LayerType::SEQUENTIAL: {
                auto* seq = reinterpret_cast<const dyn::layers::Sequential<TI>*>(layer.data);
                auto* ss = reinterpret_cast<dyn::state::Sequential<TI>*>(state.data);
                dyn::evaluate_step_sequential(device, *seq, *ss, input, output, buffer);
                break;
            }
            default:
                evaluate(device, layer, input, output, buffer);
                break;
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif

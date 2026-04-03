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
        if(total_bytes > 0) tensor.data = new char[total_bytes];
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, dyn::Tensor<SPEC>& tensor){
        if(tensor.data != nullptr){ delete[] reinterpret_cast<char*>(tensor.data); tensor.data = nullptr; }
    }

    namespace dyn{
        // --- Tensor helpers ---
        template <typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void set_shape(Tensor<TensorSpecification<TI>>& tensor, TI rank, const TI* shape){
            tensor.rank = rank; tensor.size = 1;
            for(TI i = 0; i < rank; i++){ tensor.shape[i] = shape[i]; tensor.size *= shape[i]; }
        }
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT float get(DEVICE& device, const Tensor<TensorSpecification<TI>>& tensor, TI flat_index){
            return to_float(reinterpret_cast<const char*>(tensor.data) + flat_index * size_of<TI>(tensor.type), tensor.type);
        }
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void set(DEVICE& device, Tensor<TensorSpecification<TI>>& tensor, TI flat_index, float value){
            from_float(reinterpret_cast<char*>(tensor.data) + flat_index * size_of<TI>(tensor.type), value, tensor.type);
        }

        // --- Activation ---
        template <typename MATH_DEVICE>
        RL_TOOLS_FUNCTION_PLACEMENT inline float apply_activation(MATH_DEVICE& math_dev, ActivationFunction af, float x){
            switch(af){
                case ActivationFunction::IDENTITY: return x;
                case ActivationFunction::RELU: return math::max(math_dev, x, 0.0f);
                case ActivationFunction::GELU: { float a = math::FRAC_2_SQRTPI<float> * math::SQRT1_2<float> * 0.5f; return 0.5f * (x + x * math::tanh(math_dev, a * (0.044715f * x * x * x + x))); }
                case ActivationFunction::TANH: return math::tanh(math_dev, x);
                case ActivationFunction::FAST_TANH: return math::fast_tanh(math_dev, x);
                case ActivationFunction::SIGMOID: return 1.0f / (1.0f + math::exp(math_dev, -x));
                default: return x;
            }
        }

        // --- Shape propagation ---
        template <typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void propagate_shapes(Layer<TI>& layer, const TI* in_shape, TI in_rank, TI in_size){
            switch(layer.type){
                case LayerType::DENSE: {
                    auto& d = layer.template as<layers::Dense<TI>>();
                    layer.output_rank = in_rank; layer.output_size = 1;
                    for(TI i = 0; i < in_rank - 1; i++){ layer.output_shape[i] = in_shape[i]; layer.output_size *= in_shape[i]; }
                    layer.output_shape[in_rank - 1] = d.output_dim; layer.output_size *= d.output_dim;
                    break;
                }
                case LayerType::GRU: {
                    auto& g = layer.template as<layers::GRU<TI>>();
                    layer.output_rank = in_rank; layer.output_size = 1;
                    for(TI i = 0; i < in_rank - 1; i++){ layer.output_shape[i] = in_shape[i]; layer.output_size *= in_shape[i]; }
                    layer.output_shape[in_rank - 1] = g.hidden_dim; layer.output_size *= g.hidden_dim;
                    break;
                }
                case LayerType::CONV2D: {
                    auto& c = layer.template as<layers::Conv2d<TI>>();
                    TI ih = in_shape[in_rank-3], iw = in_shape[in_rank-2];
                    TI oh = (ih + 2*c.padding_h - c.kernel_height) / c.stride_h + 1;
                    TI ow = (iw + 2*c.padding_w - c.kernel_width) / c.stride_w + 1;
                    TI batch = in_size / (ih * iw * c.input_channels);
                    layer.output_rank = 4; layer.output_shape[0] = batch; layer.output_shape[1] = oh; layer.output_shape[2] = ow; layer.output_shape[3] = c.output_channels;
                    layer.output_size = batch * oh * ow * c.output_channels;
                    break;
                }
                case LayerType::MAX_POOL2D: {
                    auto& mp = layer.template as<layers::MaxPool2d<TI>>();
                    TI ih = in_shape[in_rank-3], iw = in_shape[in_rank-2], ch = in_shape[in_rank-1];
                    TI oh = (ih + 2*mp.padding_h - mp.kernel_height) / mp.stride_h + 1;
                    TI ow = (iw + 2*mp.padding_w - mp.kernel_width) / mp.stride_w + 1;
                    TI batch = in_size / (ih * iw * ch);
                    layer.output_rank = 4; layer.output_shape[0] = batch; layer.output_shape[1] = oh; layer.output_shape[2] = ow; layer.output_shape[3] = ch;
                    layer.output_size = batch * oh * ow * ch;
                    break;
                }
                case LayerType::AVG_POOL2D: {
                    TI ch = in_shape[in_rank-1], ih = in_shape[in_rank-3], iw = in_shape[in_rank-2];
                    TI batch = in_size / (ih * iw * ch);
                    layer.output_rank = 2; layer.output_shape[0] = batch; layer.output_shape[1] = ch; layer.output_size = batch * ch;
                    break;
                }
                case LayerType::FLATTEN: {
                    if(in_rank >= 3){ TI flat = 1; for(TI i = in_rank-3; i < in_rank; i++) flat *= in_shape[i]; TI batch = in_size / flat; layer.output_rank = 2; layer.output_shape[0] = batch; layer.output_shape[1] = flat; }
                    else{ layer.output_rank = in_rank; for(TI i = 0; i < in_rank; i++) layer.output_shape[i] = in_shape[i]; }
                    layer.output_size = in_size;
                    break;
                }
                case LayerType::SAMPLE_AND_SQUASH: {
                    TI last = in_shape[in_rank-1], batch = in_size / last;
                    layer.output_rank = 2; layer.output_shape[0] = batch; layer.output_shape[1] = last/2; layer.output_size = batch * (last/2);
                    break;
                }
                case LayerType::STANDARDIZE:
                case LayerType::UNFLATTEN:
                default: {
                    layer.output_rank = in_rank; layer.output_size = in_size;
                    for(TI i = 0; i < in_rank; i++) layer.output_shape[i] = in_shape[i];
                    break;
                }
                case LayerType::SEQUENTIAL:
                case LayerType::MLP:
                case LayerType::RESNET_BLOCK:
                case LayerType::PARALLEL: {
                    // Chain shapes through children
                    const TI* cur_shape = in_shape; TI cur_rank = in_rank, cur_size = in_size;
                    for(TI i = 0; i < layer.num_children; i++){
                        propagate_shapes(layer.children[i], cur_shape, cur_rank, cur_size);
                        cur_shape = layer.children[i].output_shape;
                        cur_rank = layer.children[i].output_rank;
                        cur_size = layer.children[i].output_size;
                    }
                    // For ResnetBlock, also propagate downsample (last child) from INPUT not from conv2
                    if(layer.type == LayerType::RESNET_BLOCK && layer.num_children == 3){
                        propagate_shapes(layer.children[2], in_shape, in_rank, in_size);
                    }
                    // Output = last sequential child's output (conv2 for resnet, last layer for sequential/mlp)
                    TI last = (layer.type == LayerType::RESNET_BLOCK) ? 1 : layer.num_children - 1;
                    layer.output_rank = layer.children[last].output_rank;
                    layer.output_size = layer.children[last].output_size;
                    for(TI i = 0; i < layer.output_rank; i++) layer.output_shape[i] = layer.children[last].output_shape[i];
                    break;
                }
            }
        }

        // --- Buffer sizing (no switch needed for composites — just recurse children) ---
        template <typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT TI max_output_size(const Layer<TI>& layer){
            TI m = layer.output_size;
            for(TI i = 0; i < layer.num_children; i++){ TI s = max_output_size(layer.children[i]); if(s > m) m = s; }
            return m;
        }
        template <typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT TI max_scratch_size(const Layer<TI>& layer){
            TI m = 0;
            // Leaf-specific scratch needs
            switch(layer.type){
                case LayerType::GRU: {
                    auto& g = layer.template as<const layers::GRU<TI>>();
                    TI batch = (layer.output_rank == 3) ? layer.output_shape[1] : (layer.output_size / g.hidden_dim);
                    m = batch * 3 * g.hidden_dim;
                    break;
                }
                case LayerType::RESNET_BLOCK: {
                    m = layer.children[0].output_size; // conv1 intermediate
                    TI child_max = 0;
                    for(TI i = 0; i < layer.num_children; i++){ TI s = max_scratch_size(layer.children[i]); if(s > child_max) child_max = s; }
                    m += child_max;
                    return m;
                }
                default: break;
            }
            // Recurse into children
            for(TI i = 0; i < layer.num_children; i++){ TI s = max_scratch_size(layer.children[i]); if(s > m) m = s; }
            return m;
        }

        // --- Leaf evaluate helpers ---
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_dense(DEVICE& device, const layers::Dense<TI>& layer, const Tensor<TensorSpecification<TI>>& input, Tensor<TensorSpecification<TI>>& output){
            TI batch_size = input.size / layer.input_dim;
            for(TI b = 0; b < batch_size; b++)
                for(TI o = 0; o < layer.output_dim; o++){
                    float acc = get(device, layer.biases, o);
                    for(TI i = 0; i < layer.input_dim; i++) acc += get(device, layer.weights, o * layer.input_dim + i) * get(device, input, b * layer.input_dim + i);
                    set(device, output, b * layer.output_dim + o, apply_activation(device.math, layer.activation_function, acc));
                }
        }
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step_gru(DEVICE& device, const layers::GRU<TI>& layer, const Tensor<TensorSpecification<TI>>& input, state::GRU<TI>& gru_state, Tensor<TensorSpecification<TI>>& scratch){
            TI batch_size = input.size / layer.input_dim, hidden_dim = layer.hidden_dim;
            if(!gru_state.initialized){
                for(TI b = 0; b < batch_size; b++) for(TI h = 0; h < hidden_dim; h++) set(device, gru_state.hidden, b*hidden_dim+h, get(device, layer.initial_hidden_state, h));
                gru_state.initialized = true;
            }
            for(TI b = 0; b < batch_size; b++){
                for(TI g = 0; g < 2*hidden_dim; g++){
                    float wh = get(device, layer.biases_hidden, g);
                    for(TI h = 0; h < hidden_dim; h++) wh += get(device, layer.weights_hidden, g*hidden_dim+h) * get(device, gru_state.hidden, b*hidden_dim+h);
                    float wi = get(device, layer.biases_input, g);
                    for(TI i = 0; i < layer.input_dim; i++) wi += get(device, layer.weights_input, g*layer.input_dim+i) * get(device, input, b*layer.input_dim+i);
                    set(device, scratch, b*2*hidden_dim+g, 1.0f / (1.0f + math::exp(device.math, -(wh+wi))));
                }
                for(TI h = 0; h < hidden_dim; h++){
                    TI g = 2*hidden_dim+h;
                    float wh_n = get(device, layer.biases_hidden, g);
                    for(TI hh = 0; hh < hidden_dim; hh++) wh_n += get(device, layer.weights_hidden, g*hidden_dim+hh) * get(device, gru_state.hidden, b*hidden_dim+hh);
                    float wi_n = get(device, layer.biases_input, g);
                    for(TI i = 0; i < layer.input_dim; i++) wi_n += get(device, layer.weights_input, g*layer.input_dim+i) * get(device, input, b*layer.input_dim+i);
                    float r = get(device, scratch, b*2*hidden_dim+h);
                    float n_pre = r * wh_n + wi_n;
                    float n = math::tanh(device.math, n_pre);
                    float z = get(device, scratch, b*2*hidden_dim+hidden_dim+h);
                    set(device, scratch, b*2*hidden_dim+h, z * get(device, gru_state.hidden, b*hidden_dim+h) + (1.0f-z) * n);
                }
                for(TI h = 0; h < hidden_dim; h++) set(device, gru_state.hidden, b*hidden_dim+h, get(device, scratch, b*2*hidden_dim+h));
            }
        }
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_conv2d(DEVICE& device, const layers::Conv2d<TI>& layer, const Tensor<TensorSpecification<TI>>& input, Tensor<TensorSpecification<TI>>& output){
            TI ih = input.shape[input.rank-3], iw = input.shape[input.rank-2];
            TI batch = input.size / (ih * iw * layer.input_channels);
            TI oh = (ih + 2*layer.padding_h - layer.kernel_height) / layer.stride_h + 1;
            TI ow = (iw + 2*layer.padding_w - layer.kernel_width) / layer.stride_w + 1;
            for(TI bi = 0; bi < batch; bi++)
                for(TI ohi = 0; ohi < oh; ohi++)
                    for(TI owi = 0; owi < ow; owi++)
                        for(TI oc = 0; oc < layer.output_channels; oc++){
                            float acc = get(device, layer.biases, oc);
                            for(TI kh = 0; kh < layer.kernel_height; kh++){
                                TI ihi = ohi*layer.stride_h+kh-layer.padding_h; if(ihi >= ih) continue;
                                for(TI kw = 0; kw < layer.kernel_width; kw++){
                                    TI iwi = owi*layer.stride_w+kw-layer.padding_w; if(iwi >= iw) continue;
                                    for(TI ic = 0; ic < layer.input_channels; ic++)
                                        acc += get(device, layer.weights, ((oc*layer.kernel_height+kh)*layer.kernel_width+kw)*layer.input_channels+ic)
                                             * get(device, input, ((bi*ih+ihi)*iw+iwi)*layer.input_channels+ic);
                                }
                            }
                            set(device, output, ((bi*oh+ohi)*ow+owi)*layer.output_channels+oc, acc);
                        }
            if(layer.normalization == layers::Conv2d<TI>::Normalization::BATCH_NORM){
                float eps = 1e-5f;
                for(TI i = 0; i < output.size; i++){
                    TI oc = i % layer.output_channels;
                    set(device, output, i, get(device, layer.gamma, oc) * (get(device, output, i) - get(device, layer.running_mean, oc)) / math::sqrt(device.math, get(device, layer.running_var, oc) + eps) + get(device, layer.beta, oc));
                }
            }
            if(layer.activation_function != ActivationFunction::IDENTITY)
                for(TI i = 0; i < output.size; i++) set(device, output, i, apply_activation(device.math, layer.activation_function, get(device, output, i)));
        }
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_max_pool2d(DEVICE& device, const layers::MaxPool2d<TI>& layer, const Tensor<TensorSpecification<TI>>& input, Tensor<TensorSpecification<TI>>& output){
            TI ih = input.shape[input.rank-3], iw = input.shape[input.rank-2], ch = input.shape[input.rank-1];
            TI batch = input.size / (ih*iw*ch);
            TI oh = (ih + 2*layer.padding_h - layer.kernel_height) / layer.stride_h + 1;
            TI ow = (iw + 2*layer.padding_w - layer.kernel_width) / layer.stride_w + 1;
            for(TI bi = 0; bi < batch; bi++)
                for(TI ohi = 0; ohi < oh; ohi++)
                    for(TI owi = 0; owi < ow; owi++)
                        for(TI c = 0; c < ch; c++){
                            float mx = -1e30f;
                            for(TI kh = 0; kh < layer.kernel_height; kh++){
                                TI ihi = ohi*layer.stride_h+kh-layer.padding_h; if(ihi >= ih) continue;
                                for(TI kw = 0; kw < layer.kernel_width; kw++){
                                    TI iwi = owi*layer.stride_w+kw-layer.padding_w; if(iwi >= iw) continue;
                                    float v = get(device, input, ((bi*ih+ihi)*iw+iwi)*ch+c); if(v > mx) mx = v;
                                }
                            }
                            set(device, output, ((bi*oh+ohi)*ow+owi)*ch+c, mx);
                        }
        }
    }

    // Forward declarations
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, const dyn::Layer<TI>& layer, const dyn::Tensor<dyn::TensorSpecification<TI>>& input, dyn::Tensor<dyn::TensorSpecification<TI>>& output, dyn::Buffer<TI>& buffer);
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step(DEVICE& device, const dyn::Layer<TI>& layer, const dyn::Tensor<dyn::TensorSpecification<TI>>& input, dyn::State<TI>& state, dyn::Tensor<dyn::TensorSpecification<TI>>& output, dyn::Buffer<TI>& buffer);

    namespace dyn{
        // --- Composite evaluate: sequential chain through children ---
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_chain(DEVICE& device, const Layer<TI>& layer, const Tensor<TensorSpecification<TI>>& input, Tensor<TensorSpecification<TI>>& output, Buffer<TI>& buffer){
            const Tensor<TensorSpecification<TI>>* current_input = &input;
            for(TI i = 0; i < layer.num_children; i++){
                Tensor<TensorSpecification<TI>>* current_output;
                if(i == layer.num_children - 1) current_output = &output;
                else current_output = (current_input == &buffer.tick) ? &buffer.tock : &buffer.tick;
                rl_tools::evaluate(device, layer.children[i], *current_input, *current_output, buffer);
                current_input = current_output;
            }
        }
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step_chain(DEVICE& device, const Layer<TI>& layer, state::Composite<TI>& comp_state, const Tensor<TensorSpecification<TI>>& input, Tensor<TensorSpecification<TI>>& output, Buffer<TI>& buffer){
            const Tensor<TensorSpecification<TI>>* current_input = &input;
            for(TI i = 0; i < layer.num_children; i++){
                Tensor<TensorSpecification<TI>>* current_output;
                if(i == layer.num_children - 1) current_output = &output;
                else current_output = (current_input == &buffer.tick) ? &buffer.tock : &buffer.tick;
                rl_tools::evaluate_step(device, layer.children[i], *current_input, comp_state.child_states[i], *current_output, buffer);
                current_input = current_output;
            }
        }
        // --- GRU evaluate (full sequence) ---
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_gru(DEVICE& device, const layers::GRU<TI>& gru, const Layer<TI>& layer, const Tensor<TensorSpecification<TI>>& input, Tensor<TensorSpecification<TI>>& output, Buffer<TI>& buffer){
            TI hidden_dim = gru.hidden_dim;
            TI batch = (input.rank == 3) ? input.shape[1] : (input.size / gru.input_dim);
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
                evaluate_step_gru(device, gru, input, tmp_state, gate_scratch);
                for(TI b = 0; b < batch; b++) for(TI h = 0; h < hidden_dim; h++) set(device, output, b*hidden_dim+h, get(device, tmp_state.hidden, b*hidden_dim+h));
            }
            else{
                TI seq_len = input.shape[0];
                Tensor<TensorSpecification<TI>> step_input;
                TI step_shape[] = {batch, gru.input_dim};
                set_shape(step_input, (TI)2, step_shape);
                step_input.type = input.type;
                for(TI t = 0; t < seq_len; t++){
                    step_input.data = reinterpret_cast<char*>(const_cast<void*>(input.data)) + t * batch * gru.input_dim * size_of<TI>(input.type);
                    evaluate_step_gru(device, gru, step_input, tmp_state, gate_scratch);
                    for(TI b = 0; b < batch; b++) for(TI h = 0; h < hidden_dim; h++) set(device, output, (t*batch+b)*hidden_dim+h, get(device, tmp_state.hidden, b*hidden_dim+h));
                }
            }
        }
    }

    // --- Buffer malloc/free ---
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, dyn::Buffer<TI>& buffer){
        TI tt_size = dyn::max_output_size(*buffer.layer);
        TI sc_size = dyn::max_scratch_size(*buffer.layer);
        TI tt[] = {tt_size}; dyn::set_shape(buffer.tick, (TI)1, tt); buffer.tick.type = dyn::Type::FLOAT32; rl_tools::malloc(device, buffer.tick);
        dyn::set_shape(buffer.tock, (TI)1, tt); buffer.tock.type = dyn::Type::FLOAT32; rl_tools::malloc(device, buffer.tock);
        if(sc_size > 0){ TI sc[] = {sc_size}; dyn::set_shape(buffer.scratch, (TI)1, sc); buffer.scratch.type = dyn::Type::FLOAT32; rl_tools::malloc(device, buffer.scratch); }
    }
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, dyn::Buffer<TI>& buffer){
        rl_tools::free(device, buffer.tick); rl_tools::free(device, buffer.tock); rl_tools::free(device, buffer.scratch);
    }

    // --- State malloc/free/reset ---
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, dyn::State<TI>& state){
        state.type = state.layer->type;
        if(state.type == dyn::LayerType::GRU){
            auto& g = state.layer->template as<const dyn::layers::GRU<TI>>();
            auto* gs = new dyn::state::GRU<TI>();
            TI shape[] = {state.batch_size, g.hidden_dim};
            dyn::set_shape(gs->hidden, (TI)2, shape); gs->hidden.type = dyn::Type::FLOAT32;
            rl_tools::malloc(device, gs->hidden); gs->initialized = false;
            state.data = gs;
        }
        else if(state.layer->num_children > 0){
            auto* cs = new dyn::state::Composite<TI>();
            cs->num_children = state.layer->num_children;
            cs->child_states = new dyn::State<TI>[cs->num_children];
            for(TI i = 0; i < cs->num_children; i++){
                cs->child_states[i].batch_size = state.batch_size;
                cs->child_states[i].layer = &state.layer->children[i];
                rl_tools::malloc(device, cs->child_states[i]);
            }
            state.data = cs;
        }
        else{ state.data = nullptr; }
    }
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, dyn::State<TI>& state){
        if(state.type == dyn::LayerType::GRU){
            auto* gs = reinterpret_cast<dyn::state::GRU<TI>*>(state.data);
            rl_tools::free(device, gs->hidden); delete gs;
        }
        else if(state.data){
            auto* cs = reinterpret_cast<dyn::state::Composite<TI>*>(state.data);
            for(TI i = 0; i < cs->num_children; i++) rl_tools::free(device, cs->child_states[i]);
            delete[] cs->child_states; delete cs;
        }
        state.data = nullptr;
    }
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE& device, const dyn::Layer<TI>& layer, dyn::State<TI>& state){
        if(state.type == dyn::LayerType::GRU) reinterpret_cast<dyn::state::GRU<TI>*>(state.data)->initialized = false;
        else if(state.data){
            auto* cs = reinterpret_cast<dyn::state::Composite<TI>*>(state.data);
            for(TI i = 0; i < cs->num_children; i++) reset(device, layer.children[i], cs->child_states[i]);
        }
    }

    // --- Layer free ---
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, dyn::Layer<TI>& layer){
        // Free children first
        for(TI i = 0; i < layer.num_children; i++) rl_tools::free(device, layer.children[i]);
        if(layer.children){ delete[] layer.children; layer.children = nullptr; }
        // Free leaf data
        switch(layer.type){
            case dyn::LayerType::DENSE: { auto& d = layer.template as<dyn::layers::Dense<TI>>(); rl_tools::free(device, d.weights); rl_tools::free(device, d.biases); delete &d; break; }
            case dyn::LayerType::GRU: { auto& g = layer.template as<dyn::layers::GRU<TI>>(); rl_tools::free(device, g.weights_input); rl_tools::free(device, g.biases_input); rl_tools::free(device, g.weights_hidden); rl_tools::free(device, g.biases_hidden); rl_tools::free(device, g.initial_hidden_state); delete &g; break; }
            case dyn::LayerType::CONV2D: { auto& c = layer.template as<dyn::layers::Conv2d<TI>>(); rl_tools::free(device, c.weights); rl_tools::free(device, c.biases); if(c.normalization != dyn::layers::Conv2d<TI>::Normalization::NONE){ rl_tools::free(device, c.gamma); rl_tools::free(device, c.beta); if(c.normalization == dyn::layers::Conv2d<TI>::Normalization::BATCH_NORM){ rl_tools::free(device, c.running_mean); rl_tools::free(device, c.running_var); } } delete &c; break; }
            case dyn::LayerType::MAX_POOL2D: delete reinterpret_cast<dyn::layers::MaxPool2d<TI>*>(layer.data); break;
            case dyn::LayerType::STANDARDIZE: { auto& s = layer.template as<dyn::layers::Standardize<TI>>(); rl_tools::free(device, s.mean); rl_tools::free(device, s.precision); delete &s; break; }
            case dyn::LayerType::EMBEDDING: { auto& e = layer.template as<dyn::layers::Embedding<TI>>(); rl_tools::free(device, e.weights); delete &e; break; }
            default: break; // composites have no data, parameterless layers have nullptr
        }
        layer.data = nullptr;
    }

    // --- Top-level evaluate ---
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, const dyn::Layer<TI>& layer, const dyn::Tensor<dyn::TensorSpecification<TI>>& input, dyn::Tensor<dyn::TensorSpecification<TI>>& output, dyn::Buffer<TI>& buffer){
        dyn::set_shape(output, layer.output_rank, layer.output_shape);
        output.type = dyn::Type::FLOAT32;
        switch(layer.type){
            case dyn::LayerType::DENSE: dyn::evaluate_dense(device, layer.template as<const dyn::layers::Dense<TI>>(), input, output); break;
            case dyn::LayerType::GRU: dyn::evaluate_gru(device, layer.template as<const dyn::layers::GRU<TI>>(), layer, input, output, buffer); break;
            case dyn::LayerType::CONV2D: dyn::evaluate_conv2d(device, layer.template as<const dyn::layers::Conv2d<TI>>(), input, output); break;
            case dyn::LayerType::MAX_POOL2D: dyn::evaluate_max_pool2d(device, layer.template as<const dyn::layers::MaxPool2d<TI>>(), input, output); break;
            case dyn::LayerType::SAMPLE_AND_SQUASH: {
                TI last = input.shape[input.rank-1], half = last/2, batch = input.size/last;
                for(TI b = 0; b < batch; b++) for(TI i = 0; i < half; i++) dyn::set(device, output, b*half+i, math::tanh(device.math, dyn::get(device, input, b*last+i)));
                break;
            }
            case dyn::LayerType::STANDARDIZE: {
                auto& s = layer.template as<const dyn::layers::Standardize<TI>>();
                TI batch = input.size / s.dim;
                for(TI b = 0; b < batch; b++) for(TI i = 0; i < s.dim; i++) dyn::set(device, output, b*s.dim+i, (dyn::get(device, input, b*s.dim+i) - dyn::get(device, s.mean, i)) * dyn::get(device, s.precision, i));
                break;
            }
            case dyn::LayerType::FLATTEN: case dyn::LayerType::UNFLATTEN: {
                output.type = input.type;
                TI bytes = input.size * dyn::size_of<TI>(input.type);
                const char* src = reinterpret_cast<const char*>(input.data); char* dst = reinterpret_cast<char*>(output.data);
                for(TI i = 0; i < bytes; i++) dst[i] = src[i];
                break;
            }
            case dyn::LayerType::AVG_POOL2D: {
                TI ch = input.shape[input.rank-1], w = input.shape[input.rank-2], h = input.shape[input.rank-3], batch = input.size/(h*w*ch);
                float scale = 1.0f / (float)(h*w);
                for(TI b = 0; b < batch; b++) for(TI c = 0; c < ch; c++){
                    float sum = 0; for(TI hi = 0; hi < h; hi++) for(TI wi = 0; wi < w; wi++) sum += dyn::get(device, input, ((b*h+hi)*w+wi)*ch+c);
                    dyn::set(device, output, b*ch+c, sum*scale);
                }
                break;
            }
            case dyn::LayerType::SEQUENTIAL: case dyn::LayerType::MLP: dyn::evaluate_chain(device, layer, input, output, buffer); break;
            case dyn::LayerType::RESNET_BLOCK: {
                // children[0]=conv1, children[1]=conv2, children[2]=downsample (optional)
                rl_tools::evaluate(device, layer.children[0], input, buffer.scratch, buffer);
                rl_tools::evaluate(device, layer.children[1], buffer.scratch, output, buffer);
                if(layer.num_children == 3){
                    dyn::Buffer<TI> sub_buffer = buffer;
                    TI conv1_bytes = layer.children[0].output_size * sizeof(float);
                    sub_buffer.scratch.data = reinterpret_cast<char*>(buffer.scratch.data) + conv1_bytes;
                    sub_buffer.scratch.size = buffer.scratch.size - layer.children[0].output_size;
                    rl_tools::evaluate(device, layer.children[2], input, buffer.scratch, sub_buffer);
                }
                const dyn::Tensor<dyn::TensorSpecification<TI>>& shortcut = (layer.num_children == 3) ? buffer.scratch : input;
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
                auto& g = layer.template as<const dyn::layers::GRU<TI>>();
                auto* gs = reinterpret_cast<dyn::state::GRU<TI>*>(state.data);
                dyn::set_shape(output, layer.output_rank, layer.output_shape); output.type = dyn::Type::FLOAT32;
                dyn::Tensor<dyn::TensorSpecification<TI>> gate_scratch;
                TI batch = input.size / g.input_dim;
                TI gs_shape[] = {batch * 2 * g.hidden_dim};
                dyn::set_shape(gate_scratch, (TI)1, gs_shape); gate_scratch.type = dyn::Type::FLOAT32; gate_scratch.data = buffer.scratch.data;
                dyn::evaluate_step_gru(device, g, input, *gs, gate_scratch);
                for(TI b = 0; b < batch; b++) for(TI h = 0; h < g.hidden_dim; h++) dyn::set(device, output, b*g.hidden_dim+h, dyn::get(device, gs->hidden, b*g.hidden_dim+h));
                break;
            }
            case dyn::LayerType::SEQUENTIAL: case dyn::LayerType::MLP: {
                auto* cs = reinterpret_cast<dyn::state::Composite<TI>*>(state.data);
                dyn::evaluate_step_chain(device, layer, *cs, input, output, buffer);
                break;
            }
            default: evaluate(device, layer, input, output, buffer); break;
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif

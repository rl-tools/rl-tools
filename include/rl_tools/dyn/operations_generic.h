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
    // --- Tensor operations ---
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
            TI element_size = dyn::size_of<TI>(tensor.type);
            const char* ptr = reinterpret_cast<const char*>(tensor.data) + flat_index * element_size;
            return dyn::to_float(ptr, tensor.type);
        }
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void set(DEVICE& device, Tensor<TensorSpecification<TI>>& tensor, TI flat_index, float value){
            TI element_size = dyn::size_of<TI>(tensor.type);
            char* ptr = reinterpret_cast<char*>(tensor.data) + flat_index * element_size;
            dyn::from_float(ptr, value, tensor.type);
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

        template <typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT TI output_dim(const Layer<TI>& layer){
            switch(layer.type){
                case LayerType::DENSE: return reinterpret_cast<const layers::Dense<TI>*>(layer.data)->output_dim;
                case LayerType::GRU: return reinterpret_cast<const layers::GRU<TI>*>(layer.data)->hidden_dim;
                case LayerType::EMBEDDING: return reinterpret_cast<const layers::Embedding<TI>*>(layer.data)->embedding_dim;
                case LayerType::SEQUENTIAL: {
                    auto* seq = reinterpret_cast<const layers::Sequential<TI>*>(layer.data);
                    return output_dim(seq->layers[seq->num_layers - 1]);
                }
                case LayerType::MLP: return output_dim(reinterpret_cast<const layers::MLP<TI>*>(layer.data)->output_layer);
                default: return 0;
            }
        }
        template <typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT TI input_dim(const Layer<TI>& layer){
            switch(layer.type){
                case LayerType::DENSE: return reinterpret_cast<const layers::Dense<TI>*>(layer.data)->input_dim;
                case LayerType::GRU: return reinterpret_cast<const layers::GRU<TI>*>(layer.data)->input_dim;
                case LayerType::SEQUENTIAL: {
                    auto* seq = reinterpret_cast<const layers::Sequential<TI>*>(layer.data);
                    return input_dim(seq->layers[0]);
                }
                case LayerType::MLP: return input_dim(reinterpret_cast<const layers::MLP<TI>*>(layer.data)->input_layer);
                default: return 0;
            }
        }
        template <typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT TI max_intermediate_size(const Layer<TI>& layer, TI batch_size){
            switch(layer.type){
                case LayerType::DENSE: return batch_size * reinterpret_cast<const layers::Dense<TI>*>(layer.data)->output_dim;
                case LayerType::GRU: return batch_size * reinterpret_cast<const layers::GRU<TI>*>(layer.data)->hidden_dim;
                case LayerType::SEQUENTIAL: {
                    auto* seq = reinterpret_cast<const layers::Sequential<TI>*>(layer.data);
                    TI max_size = 0;
                    for(TI i = 0; i < seq->num_layers; i++){
                        TI layer_size = max_intermediate_size(seq->layers[i], batch_size);
                        if(layer_size > max_size) max_size = layer_size;
                    }
                    return max_size;
                }
                case LayerType::MLP: {
                    auto* mlp = reinterpret_cast<const layers::MLP<TI>*>(layer.data);
                    TI max_size = max_intermediate_size(mlp->input_layer, batch_size);
                    for(TI i = 0; i < mlp->num_hidden_layers; i++){
                        TI s = max_intermediate_size(mlp->hidden_layers[i], batch_size);
                        if(s > max_size) max_size = s;
                    }
                    TI s = max_intermediate_size(mlp->output_layer, batch_size);
                    if(s > max_size) max_size = s;
                    return max_size;
                }
                default: return 0;
            }
        }
        template <typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT TI max_gru_hidden_size(const Layer<TI>& layer, TI batch_size){
            TI max_size = 0;
            switch(layer.type){
                case LayerType::GRU: return batch_size * reinterpret_cast<const layers::GRU<TI>*>(layer.data)->hidden_dim;
                case LayerType::SEQUENTIAL: {
                    auto* seq = reinterpret_cast<const layers::Sequential<TI>*>(layer.data);
                    for(TI i = 0; i < seq->num_layers; i++){
                        TI s = max_gru_hidden_size(seq->layers[i], batch_size);
                        if(s > max_size) max_size = s;
                    }
                    return max_size;
                }
                case LayerType::MLP: {
                    auto* mlp = reinterpret_cast<const layers::MLP<TI>*>(layer.data);
                    TI s = max_gru_hidden_size(mlp->input_layer, batch_size);
                    if(s > max_size) max_size = s;
                    for(TI i = 0; i < mlp->num_hidden_layers; i++){
                        s = max_gru_hidden_size(mlp->hidden_layers[i], batch_size);
                        if(s > max_size) max_size = s;
                    }
                    s = max_gru_hidden_size(mlp->output_layer, batch_size);
                    if(s > max_size) max_size = s;
                    return max_size;
                }
                default: return 0;
            }
        }
        template <typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT TI max_gru_gate_size(const Layer<TI>& layer, TI batch_size){
            TI max_size = 0;
            switch(layer.type){
                case LayerType::GRU: return batch_size * 2 * reinterpret_cast<const layers::GRU<TI>*>(layer.data)->hidden_dim;
                case LayerType::SEQUENTIAL: {
                    auto* seq = reinterpret_cast<const layers::Sequential<TI>*>(layer.data);
                    for(TI i = 0; i < seq->num_layers; i++){
                        TI s = max_gru_gate_size(seq->layers[i], batch_size);
                        if(s > max_size) max_size = s;
                    }
                    return max_size;
                }
                case LayerType::MLP: {
                    auto* mlp = reinterpret_cast<const layers::MLP<TI>*>(layer.data);
                    TI s = max_gru_gate_size(mlp->input_layer, batch_size);
                    if(s > max_size) max_size = s;
                    for(TI i = 0; i < mlp->num_hidden_layers; i++){
                        s = max_gru_gate_size(mlp->hidden_layers[i], batch_size);
                        if(s > max_size) max_size = s;
                    }
                    s = max_gru_gate_size(mlp->output_layer, batch_size);
                    if(s > max_size) max_size = s;
                    return max_size;
                }
                default: return 0;
            }
        }

        // --- Dense evaluate ---
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

        // --- GRU evaluate_step ---
        // scratch layout: [batch * 2 * hidden_dim] for r,z gates, then [batch * hidden_dim] for new_h
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step_gru(DEVICE& device, const layers::GRU<TI>& layer, const Tensor<TensorSpecification<TI>>& input, state::GRU<TI>& gru_state, Tensor<TensorSpecification<TI>>& scratch){
            TI batch_size = input.size / layer.input_dim;
            TI hidden_dim = layer.hidden_dim;
            if(!gru_state.initialized){
                for(TI b = 0; b < batch_size; b++){
                    for(TI h = 0; h < hidden_dim; h++){
                        set(device, gru_state.hidden, b * hidden_dim + h, get(device, layer.initial_hidden_state, h));
                    }
                }
                gru_state.initialized = true;
            }
            for(TI b = 0; b < batch_size; b++){
                // r and z gates
                for(TI g = 0; g < 2 * hidden_dim; g++){
                    float wh = get(device, layer.biases_hidden, g);
                    for(TI h = 0; h < hidden_dim; h++){
                        wh += get(device, layer.weights_hidden, g * hidden_dim + h) * get(device, gru_state.hidden, b * hidden_dim + h);
                    }
                    float wi = get(device, layer.biases_input, g);
                    for(TI i = 0; i < layer.input_dim; i++){
                        wi += get(device, layer.weights_input, g * layer.input_dim + i) * get(device, input, b * layer.input_dim + i);
                    }
                    float rz = 1.0f / (1.0f + __builtin_expf(-(wh + wi)));
                    set(device, scratch, b * 2 * hidden_dim + g, rz);
                }
                // n gate and new hidden state (compute all, then write)
                for(TI h = 0; h < hidden_dim; h++){
                    TI g = 2 * hidden_dim + h;
                    float wh_n = get(device, layer.biases_hidden, g);
                    for(TI hh = 0; hh < hidden_dim; hh++){
                        wh_n += get(device, layer.weights_hidden, g * hidden_dim + hh) * get(device, gru_state.hidden, b * hidden_dim + hh);
                    }
                    float wi_n = get(device, layer.biases_input, g);
                    for(TI i = 0; i < layer.input_dim; i++){
                        wi_n += get(device, layer.weights_input, g * layer.input_dim + i) * get(device, input, b * layer.input_dim + i);
                    }
                    float r = get(device, scratch, b * 2 * hidden_dim + h);
                    float n_pre = r * wh_n + wi_n;
                    float n;
                    if(n_pre > 10.0f) n = 1.0f;
                    else if(n_pre < -10.0f) n = -1.0f;
                    else{
                        float e2x = __builtin_expf(2.0f * n_pre);
                        n = (e2x - 1.0f) / (e2x + 1.0f);
                    }
                    float z = get(device, scratch, b * 2 * hidden_dim + hidden_dim + h);
                    float prev_h = get(device, gru_state.hidden, b * hidden_dim + h);
                    float new_h = z * prev_h + (1.0f - z) * n;
                    // Write to scratch (after the r,z area) to avoid corrupting state mid-loop
                    // Reuse scratch at offset batch_size * 2 * hidden_dim is not safe because scratch might be too small
                    // Instead, we know the n gate reads state but doesn't modify it, so we can write directly
                    // The bug was: we write to gru_state.hidden[h] before computing n for other h values
                    // But actually the n gate computation reads gru_state.hidden[hh] for ALL hh, not just h
                    // So we MUST NOT modify gru_state.hidden until ALL n values are computed
                    // Store new_h temporarily in scratch (overwriting r values which are no longer needed)
                    set(device, scratch, b * 2 * hidden_dim + h, new_h);
                }
                // Copy computed new hidden state back
                for(TI h = 0; h < hidden_dim; h++){
                    set(device, gru_state.hidden, b * hidden_dim + h, get(device, scratch, b * 2 * hidden_dim + h));
                }
            }
        }

        // --- GRU evaluate (full sequence) ---
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_gru(DEVICE& device, const layers::GRU<TI>& layer, const Tensor<TensorSpecification<TI>>& input, Tensor<TensorSpecification<TI>>& output, Buffer<TI>& buffer){
            TI hidden_dim = layer.hidden_dim;
            if(input.rank == 2){
                TI batch_size = input.shape[0];
                state::GRU<TI> tmp_state;
                TI state_shape[] = {batch_size, hidden_dim};
                set_shape(tmp_state.hidden, (TI)2, state_shape);
                tmp_state.hidden.type = dyn::Type::FLOAT32;
                tmp_state.hidden.data = buffer.gru_state_scratch.data;
                tmp_state.initialized = false;
                Tensor<TensorSpecification<TI>> scratch;
                TI scratch_shape[] = {batch_size * 2 * hidden_dim};
                set_shape(scratch, (TI)1, scratch_shape);
                scratch.type = dyn::Type::FLOAT32;
                scratch.data = buffer.gru_gate_scratch.data;
                evaluate_step_gru(device, layer, input, tmp_state, scratch);
                for(TI b = 0; b < batch_size; b++){
                    for(TI h = 0; h < hidden_dim; h++){
                        set(device, output, b * hidden_dim + h, get(device, tmp_state.hidden, b * hidden_dim + h));
                    }
                }
            }
            else{
                TI seq_len = input.shape[0];
                TI batch_size = input.shape[1];
                state::GRU<TI> tmp_state;
                TI state_shape[] = {batch_size, hidden_dim};
                set_shape(tmp_state.hidden, (TI)2, state_shape);
                tmp_state.hidden.type = dyn::Type::FLOAT32;
                tmp_state.hidden.data = buffer.gru_state_scratch.data;
                tmp_state.initialized = false;
                Tensor<TensorSpecification<TI>> step_input, scratch;
                TI step_input_shape[] = {batch_size, layer.input_dim};
                set_shape(step_input, (TI)2, step_input_shape);
                step_input.type = input.type;
                TI scratch_shape[] = {batch_size * 2 * hidden_dim};
                set_shape(scratch, (TI)1, scratch_shape);
                scratch.type = dyn::Type::FLOAT32;
                scratch.data = buffer.gru_gate_scratch.data;
                for(TI t = 0; t < seq_len; t++){
                    TI input_element_size = dyn::size_of<TI>(input.type);
                    step_input.data = reinterpret_cast<char*>(const_cast<void*>(input.data)) + t * batch_size * layer.input_dim * input_element_size;
                    evaluate_step_gru(device, layer, step_input, tmp_state, scratch);
                    for(TI b = 0; b < batch_size; b++){
                        for(TI h = 0; h < hidden_dim; h++){
                            set(device, output, (t * batch_size + b) * hidden_dim + h, get(device, tmp_state.hidden, b * hidden_dim + h));
                        }
                    }
                }
            }
        }

        // --- SampleAndSquash evaluate ---
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_sample_and_squash(DEVICE& device, const Tensor<TensorSpecification<TI>>& input, Tensor<TensorSpecification<TI>>& output){
            TI last_dim = input.shape[input.rank - 1];
            TI half_dim = last_dim / 2;
            TI batch_size = input.size / last_dim;
            for(TI b = 0; b < batch_size; b++){
                for(TI i = 0; i < half_dim; i++){
                    float mean = get(device, input, b * last_dim + i);
                    float t;
                    if(mean > 10.0f) t = 1.0f;
                    else if(mean < -10.0f) t = -1.0f;
                    else{
                        float e2x = __builtin_expf(2.0f * mean);
                        t = (e2x - 1.0f) / (e2x + 1.0f);
                    }
                    set(device, output, b * half_dim + i, t);
                }
            }
        }

        // --- Standardize evaluate ---
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_standardize(DEVICE& device, const layers::Standardize<TI>& layer, const Tensor<TensorSpecification<TI>>& input, Tensor<TensorSpecification<TI>>& output){
            TI batch_size = input.size / layer.dim;
            for(TI b = 0; b < batch_size; b++){
                for(TI i = 0; i < layer.dim; i++){
                    float val = get(device, input, b * layer.dim + i);
                    float m = get(device, layer.mean, i);
                    float p = get(device, layer.precision, i);
                    set(device, output, b * layer.dim + i, (val - m) * p);
                }
            }
        }
    }

    // Forward declarations for mutual recursion
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, const dyn::Layer<TI>& layer, const dyn::Tensor<dyn::TensorSpecification<TI>>& input, dyn::Tensor<dyn::TensorSpecification<TI>>& output, dyn::Buffer<TI>& buffer);
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step(DEVICE& device, const dyn::Layer<TI>& layer, const dyn::Tensor<dyn::TensorSpecification<TI>>& input, dyn::State<TI>& state, dyn::Tensor<dyn::TensorSpecification<TI>>& output, dyn::Buffer<TI>& buffer);

    namespace dyn{
        // --- Sequential evaluate ---
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_sequential(DEVICE& device, const layers::Sequential<TI>& seq, const Tensor<TensorSpecification<TI>>& input, Tensor<TensorSpecification<TI>>& output, Buffer<TI>& buffer){
            const Tensor<TensorSpecification<TI>>* current_input = &input;
            Tensor<TensorSpecification<TI>>* current_output;
            bool input_is_external = true;
            for(TI i = 0; i < seq.num_layers; i++){
                bool is_last = (i == seq.num_layers - 1);
                if(is_last){
                    current_output = &output;
                }
                else{
                    current_output = input_is_external ? &buffer.tick : (current_input == &buffer.tick ? &buffer.tock : &buffer.tick);
                }
                rl_tools::evaluate(device, seq.layers[i], *current_input, *current_output, buffer);
                current_input = current_output;
                input_is_external = false;
            }
        }

        // --- Sequential evaluate_step ---
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step_sequential(DEVICE& device, const layers::Sequential<TI>& seq, const Tensor<TensorSpecification<TI>>& input, state::Sequential<TI>& seq_state, Tensor<TensorSpecification<TI>>& output, Buffer<TI>& buffer){
            const Tensor<TensorSpecification<TI>>* current_input = &input;
            Tensor<TensorSpecification<TI>>* current_output;
            bool input_is_external = true;
            for(TI i = 0; i < seq.num_layers; i++){
                bool is_last = (i == seq.num_layers - 1);
                if(is_last){
                    current_output = &output;
                }
                else{
                    current_output = input_is_external ? &buffer.tick : (current_input == &buffer.tick ? &buffer.tock : &buffer.tick);
                }
                rl_tools::evaluate_step(device, seq.layers[i], *current_input, seq_state.layer_states[i], *current_output, buffer);
                current_input = current_output;
                input_is_external = false;
            }
        }

        // --- MLP evaluate (delegates to sequential-style chaining) ---
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void evaluate_mlp(DEVICE& device, const layers::MLP<TI>& mlp, const Tensor<TensorSpecification<TI>>& input, Tensor<TensorSpecification<TI>>& output, Buffer<TI>& buffer){
            TI total_layers = 1 + mlp.num_hidden_layers + 1;
            const Tensor<TensorSpecification<TI>>* current_input = &input;
            Tensor<TensorSpecification<TI>>* current_output;
            bool input_is_external = true;
            for(TI i = 0; i < total_layers; i++){
                const Layer<TI>* current_layer;
                if(i == 0) current_layer = &mlp.input_layer;
                else if(i <= mlp.num_hidden_layers) current_layer = &mlp.hidden_layers[i - 1];
                else current_layer = &mlp.output_layer;
                bool is_last = (i == total_layers - 1);
                if(is_last){
                    current_output = &output;
                }
                else{
                    current_output = input_is_external ? &buffer.tick : (current_input == &buffer.tick ? &buffer.tock : &buffer.tick);
                }
                rl_tools::evaluate(device, *current_layer, *current_input, *current_output, buffer);
                current_input = current_output;
                input_is_external = false;
            }
        }
    }

    // --- Buffer malloc/free ---
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, dyn::Buffer<TI>& buffer){
        buffer.max_size = dyn::max_intermediate_size(*buffer.layer, buffer.batch_size);
        TI shape[] = {buffer.max_size};
        dyn::set_shape(buffer.tick, (TI)1, shape);
        buffer.tick.type = dyn::Type::FLOAT32;
        rl_tools::malloc(device, buffer.tick);
        dyn::set_shape(buffer.tock, (TI)1, shape);
        buffer.tock.type = dyn::Type::FLOAT32;
        rl_tools::malloc(device, buffer.tock);
        TI gru_hidden = dyn::max_gru_hidden_size(*buffer.layer, buffer.batch_size);
        if(gru_hidden > 0){
            TI hs[] = {gru_hidden};
            dyn::set_shape(buffer.gru_state_scratch, (TI)1, hs);
            buffer.gru_state_scratch.type = dyn::Type::FLOAT32;
            rl_tools::malloc(device, buffer.gru_state_scratch);
            TI gs[] = {dyn::max_gru_gate_size(*buffer.layer, buffer.batch_size)};
            dyn::set_shape(buffer.gru_gate_scratch, (TI)1, gs);
            buffer.gru_gate_scratch.type = dyn::Type::FLOAT32;
            rl_tools::malloc(device, buffer.gru_gate_scratch);
        }
    }
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, dyn::Buffer<TI>& buffer){
        rl_tools::free(device, buffer.tick);
        rl_tools::free(device, buffer.tock);
        rl_tools::free(device, buffer.gru_state_scratch);
        rl_tools::free(device, buffer.gru_gate_scratch);
    }

    // --- State malloc/free/reset ---
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, dyn::State<TI>& state){
        state.type = state.layer->type;
        switch(state.type){
            case dyn::LayerType::GRU: {
                auto* gru_layer = reinterpret_cast<const dyn::layers::GRU<TI>*>(state.layer->data);
                auto* gru_state = new dyn::state::GRU<TI>();
                TI shape[] = {state.batch_size, gru_layer->hidden_dim};
                dyn::set_shape(gru_state->hidden, (TI)2, shape);
                gru_state->hidden.type = dyn::Type::FLOAT32;
                rl_tools::malloc(device, gru_state->hidden);
                gru_state->initialized = false;
                state.data = gru_state;
                break;
            }
            case dyn::LayerType::SEQUENTIAL: {
                auto* seq_layer = reinterpret_cast<const dyn::layers::Sequential<TI>*>(state.layer->data);
                auto* seq_state = new dyn::state::Sequential<TI>();
                seq_state->num_layers = seq_layer->num_layers;
                seq_state->layer_states = new dyn::State<TI>[seq_layer->num_layers];
                for(TI i = 0; i < seq_layer->num_layers; i++){
                    seq_state->layer_states[i].batch_size = state.batch_size;
                    seq_state->layer_states[i].layer = &seq_layer->layers[i];
                    rl_tools::malloc(device, seq_state->layer_states[i]);
                }
                state.data = seq_state;
                break;
            }
            case dyn::LayerType::MLP: {
                auto* mlp_layer = reinterpret_cast<const dyn::layers::MLP<TI>*>(state.layer->data);
                auto* mlp_state = new dyn::state::MLP<TI>();
                mlp_state->input_layer_state.batch_size = state.batch_size;
                mlp_state->input_layer_state.layer = &mlp_layer->input_layer;
                rl_tools::malloc(device, mlp_state->input_layer_state);
                mlp_state->num_hidden_layers = mlp_layer->num_hidden_layers;
                mlp_state->hidden_layer_states = new dyn::State<TI>[mlp_layer->num_hidden_layers];
                for(TI i = 0; i < mlp_layer->num_hidden_layers; i++){
                    mlp_state->hidden_layer_states[i].batch_size = state.batch_size;
                    mlp_state->hidden_layer_states[i].layer = &mlp_layer->hidden_layers[i];
                    rl_tools::malloc(device, mlp_state->hidden_layer_states[i]);
                }
                mlp_state->output_layer_state.batch_size = state.batch_size;
                mlp_state->output_layer_state.layer = &mlp_layer->output_layer;
                rl_tools::malloc(device, mlp_state->output_layer_state);
                state.data = mlp_state;
                break;
            }
            default:
                state.data = nullptr;
                break;
        }
    }
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, dyn::State<TI>& state){
        switch(state.type){
            case dyn::LayerType::GRU: {
                auto* gru_state = reinterpret_cast<dyn::state::GRU<TI>*>(state.data);
                rl_tools::free(device, gru_state->hidden);
                delete gru_state;
                break;
            }
            case dyn::LayerType::SEQUENTIAL: {
                auto* seq_state = reinterpret_cast<dyn::state::Sequential<TI>*>(state.data);
                for(TI i = 0; i < seq_state->num_layers; i++){
                    rl_tools::free(device, seq_state->layer_states[i]);
                }
                delete[] seq_state->layer_states;
                delete seq_state;
                break;
            }
            case dyn::LayerType::MLP: {
                auto* mlp_state = reinterpret_cast<dyn::state::MLP<TI>*>(state.data);
                rl_tools::free(device, mlp_state->input_layer_state);
                for(TI i = 0; i < mlp_state->num_hidden_layers; i++){
                    rl_tools::free(device, mlp_state->hidden_layer_states[i]);
                }
                delete[] mlp_state->hidden_layer_states;
                rl_tools::free(device, mlp_state->output_layer_state);
                delete mlp_state;
                break;
            }
            default: break;
        }
        state.data = nullptr;
    }
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE& device, const dyn::Layer<TI>& layer, dyn::State<TI>& state){
        switch(state.type){
            case dyn::LayerType::GRU: {
                reinterpret_cast<dyn::state::GRU<TI>*>(state.data)->initialized = false;
                break;
            }
            case dyn::LayerType::SEQUENTIAL: {
                auto* seq_state = reinterpret_cast<dyn::state::Sequential<TI>*>(state.data);
                auto* seq_layer = reinterpret_cast<const dyn::layers::Sequential<TI>*>(layer.data);
                for(TI i = 0; i < seq_state->num_layers; i++){
                    reset(device, seq_layer->layers[i], seq_state->layer_states[i]);
                }
                break;
            }
            default: break;
        }
    }

    // --- Layer free ---
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, dyn::Layer<TI>& layer){
        switch(layer.type){
            case dyn::LayerType::DENSE: {
                auto* d = reinterpret_cast<dyn::layers::Dense<TI>*>(layer.data);
                rl_tools::free(device, d->weights);
                rl_tools::free(device, d->biases);
                delete d;
                break;
            }
            case dyn::LayerType::GRU: {
                auto* g = reinterpret_cast<dyn::layers::GRU<TI>*>(layer.data);
                rl_tools::free(device, g->weights_input);
                rl_tools::free(device, g->biases_input);
                rl_tools::free(device, g->weights_hidden);
                rl_tools::free(device, g->biases_hidden);
                rl_tools::free(device, g->initial_hidden_state);
                delete g;
                break;
            }
            case dyn::LayerType::CONV2D: {
                auto* c = reinterpret_cast<dyn::layers::Conv2d<TI>*>(layer.data);
                rl_tools::free(device, c->weights);
                rl_tools::free(device, c->biases);
                if(c->normalization != dyn::layers::Conv2d<TI>::Normalization::NONE){
                    rl_tools::free(device, c->gamma);
                    rl_tools::free(device, c->beta);
                    if(c->normalization == dyn::layers::Conv2d<TI>::Normalization::BATCH_NORM){
                        rl_tools::free(device, c->running_mean);
                        rl_tools::free(device, c->running_var);
                    }
                }
                delete c;
                break;
            }
            case dyn::LayerType::MAX_POOL2D: delete reinterpret_cast<dyn::layers::MaxPool2d<TI>*>(layer.data); break;
            case dyn::LayerType::STANDARDIZE: {
                auto* s = reinterpret_cast<dyn::layers::Standardize<TI>*>(layer.data);
                rl_tools::free(device, s->mean);
                rl_tools::free(device, s->precision);
                delete s;
                break;
            }
            case dyn::LayerType::EMBEDDING: {
                auto* e = reinterpret_cast<dyn::layers::Embedding<TI>*>(layer.data);
                rl_tools::free(device, e->weights);
                delete e;
                break;
            }
            case dyn::LayerType::SEQUENTIAL: {
                auto* seq = reinterpret_cast<dyn::layers::Sequential<TI>*>(layer.data);
                for(TI i = 0; i < seq->num_layers; i++){
                    rl_tools::free(device, seq->layers[i]);
                }
                delete[] seq->layers;
                delete seq;
                break;
            }
            case dyn::LayerType::MLP: {
                auto* mlp = reinterpret_cast<dyn::layers::MLP<TI>*>(layer.data);
                rl_tools::free(device, mlp->input_layer);
                for(TI i = 0; i < mlp->num_hidden_layers; i++){
                    rl_tools::free(device, mlp->hidden_layers[i]);
                }
                delete[] mlp->hidden_layers;
                rl_tools::free(device, mlp->output_layer);
                delete mlp;
                break;
            }
            case dyn::LayerType::PARALLEL: {
                auto* p = reinterpret_cast<dyn::layers::Parallel<TI>*>(layer.data);
                rl_tools::free(device, *p->pipeline_a); delete p->pipeline_a;
                rl_tools::free(device, *p->pipeline_b); delete p->pipeline_b;
                if(p->head){ rl_tools::free(device, *p->head); delete p->head; }
                delete p;
                break;
            }
            default: break;
        }
        layer.data = nullptr;
    }

    // --- Top-level evaluate ---
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, const dyn::Layer<TI>& layer, const dyn::Tensor<dyn::TensorSpecification<TI>>& input, dyn::Tensor<dyn::TensorSpecification<TI>>& output, dyn::Buffer<TI>& buffer){
        switch(layer.type){
            case dyn::LayerType::DENSE: {
                auto* d = reinterpret_cast<const dyn::layers::Dense<TI>*>(layer.data);
                // Preserve input rank: replace only the last dimension with output_dim
                TI out_rank = input.rank;
                TI out_shape[dyn::TensorSpecification<TI>::MAX_RANK];
                for(TI i = 0; i < input.rank; i++) out_shape[i] = input.shape[i];
                out_shape[out_rank - 1] = d->output_dim;
                dyn::set_shape(output, out_rank, out_shape);
                output.type = dyn::Type::FLOAT32;
                dyn::evaluate_dense(device, *d, input, output);
                break;
            }
            case dyn::LayerType::GRU: {
                auto* g = reinterpret_cast<const dyn::layers::GRU<TI>*>(layer.data);
                if(input.rank == 3){
                    TI out_shape[] = {input.shape[0], input.shape[1], g->hidden_dim};
                    dyn::set_shape(output, (TI)3, out_shape);
                }
                else{
                    TI batch_size = input.size / g->input_dim;
                    TI out_shape[] = {batch_size, g->hidden_dim};
                    dyn::set_shape(output, (TI)2, out_shape);
                }
                output.type = dyn::Type::FLOAT32;
                dyn::evaluate_gru(device, *g, input, output, buffer);
                break;
            }
            case dyn::LayerType::SAMPLE_AND_SQUASH: {
                TI last_dim = input.shape[input.rank - 1];
                TI half_dim = last_dim / 2;
                TI batch_size = input.size / last_dim;
                TI out_shape[] = {batch_size, half_dim};
                dyn::set_shape(output, (TI)2, out_shape);
                output.type = dyn::Type::FLOAT32;
                dyn::evaluate_sample_and_squash(device, input, output);
                break;
            }
            case dyn::LayerType::STANDARDIZE: {
                auto* s = reinterpret_cast<const dyn::layers::Standardize<TI>*>(layer.data);
                dyn::set_shape(output, input.rank, input.shape);
                output.type = dyn::Type::FLOAT32;
                dyn::evaluate_standardize(device, *s, input, output);
                break;
            }
            case dyn::LayerType::SEQUENTIAL: {
                dyn::evaluate_sequential(device, *reinterpret_cast<const dyn::layers::Sequential<TI>*>(layer.data), input, output, buffer);
                break;
            }
            case dyn::LayerType::MLP: {
                dyn::evaluate_mlp(device, *reinterpret_cast<const dyn::layers::MLP<TI>*>(layer.data), input, output, buffer);
                break;
            }
            case dyn::LayerType::FLATTEN: {
                if(input.rank >= 3){
                    TI flat_dim = 1;
                    for(TI i = input.rank - 3; i < input.rank; i++) flat_dim *= input.shape[i];
                    TI batch_size = input.size / flat_dim;
                    TI out_shape[] = {batch_size, flat_dim};
                    dyn::set_shape(output, (TI)2, out_shape);
                }
                else{
                    dyn::set_shape(output, input.rank, input.shape);
                }
                output.type = input.type;
                TI element_size = dyn::size_of<TI>(input.type);
                const char* src = reinterpret_cast<const char*>(input.data);
                char* dst = reinterpret_cast<char*>(output.data);
                for(TI i = 0; i < input.size * element_size; i++) dst[i] = src[i];
                break;
            }
            case dyn::LayerType::AVG_POOL2D: {
                TI channels = input.shape[input.rank - 1];
                TI width = input.shape[input.rank - 2];
                TI height = input.shape[input.rank - 3];
                TI batch_size = input.size / (height * width * channels);
                TI out_shape[] = {batch_size, channels};
                dyn::set_shape(output, (TI)2, out_shape);
                output.type = dyn::Type::FLOAT32;
                float scale = 1.0f / (float)(height * width);
                for(TI b = 0; b < batch_size; b++){
                    for(TI c = 0; c < channels; c++){
                        float sum = 0;
                        for(TI h = 0; h < height; h++){
                            for(TI w = 0; w < width; w++){
                                sum += dyn::get(device, input, ((b * height + h) * width + w) * channels + c);
                            }
                        }
                        dyn::set(device, output, b * channels + c, sum * scale);
                    }
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
                TI batch_size = input.size / g->input_dim;
                TI out_shape[] = {batch_size, g->hidden_dim};
                dyn::set_shape(output, (TI)2, out_shape);
                output.type = dyn::Type::FLOAT32;
                dyn::Tensor<dyn::TensorSpecification<TI>> scratch;
                TI scratch_shape[] = {batch_size * 2 * g->hidden_dim};
                dyn::set_shape(scratch, (TI)1, scratch_shape);
                scratch.type = dyn::Type::FLOAT32;
                scratch.data = buffer.gru_gate_scratch.data;
                dyn::evaluate_step_gru(device, *g, input, *gru_state, scratch);
                for(TI b = 0; b < batch_size; b++){
                    for(TI h = 0; h < g->hidden_dim; h++){
                        dyn::set(device, output, b * g->hidden_dim + h, dyn::get(device, gru_state->hidden, b * g->hidden_dim + h));
                    }
                }
                break;
            }
            case dyn::LayerType::SEQUENTIAL: {
                auto* seq = reinterpret_cast<const dyn::layers::Sequential<TI>*>(layer.data);
                auto* seq_state = reinterpret_cast<dyn::state::Sequential<TI>*>(state.data);
                dyn::evaluate_step_sequential(device, *seq, input, *seq_state, output, buffer);
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

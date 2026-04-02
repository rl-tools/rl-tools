#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_DYN_MODEL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_DYN_MODEL_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::dyn{
    enum class Type { FLOAT32, FLOAT64, BF16, INT8 };

    template <typename T_TI>
    RL_TOOLS_FUNCTION_PLACEMENT T_TI size_of(Type t){
        switch(t){
            case Type::FLOAT32: return 4;
            case Type::FLOAT64: return 8;
            case Type::BF16: return 2;
            case Type::INT8: return 1;
            default: return 0;
        }
    }
    RL_TOOLS_FUNCTION_PLACEMENT inline float to_float(const void* data, Type t){
        switch(t){
            case Type::FLOAT32: return *reinterpret_cast<const float*>(data);
            case Type::FLOAT64: return static_cast<float>(*reinterpret_cast<const double*>(data));
            case Type::BF16: {
                unsigned short bits = *reinterpret_cast<const unsigned short*>(data);
                unsigned int float_bits = static_cast<unsigned int>(bits) << 16;
                float result;
                char* result_ptr = reinterpret_cast<char*>(&result);
                const char* bits_ptr = reinterpret_cast<const char*>(&float_bits);
                for(int i = 0; i < 4; i++){
                    result_ptr[i] = bits_ptr[i];
                }
                return result;
            }
            case Type::INT8: return static_cast<float>(*reinterpret_cast<const signed char*>(data));
            default: return 0;
        }
    }
    RL_TOOLS_FUNCTION_PLACEMENT inline void from_float(void* data, float value, Type t){
        switch(t){
            case Type::FLOAT32: *reinterpret_cast<float*>(data) = value; break;
            case Type::FLOAT64: *reinterpret_cast<double*>(data) = static_cast<double>(value); break;
            case Type::BF16: {
                unsigned int float_bits;
                const char* value_ptr = reinterpret_cast<const char*>(&value);
                char* bits_ptr = reinterpret_cast<char*>(&float_bits);
                for(int i = 0; i < 4; i++){
                    bits_ptr[i] = value_ptr[i];
                }
                *reinterpret_cast<unsigned short*>(data) = static_cast<unsigned short>(float_bits >> 16);
                break;
            }
            case Type::INT8: *reinterpret_cast<signed char*>(data) = static_cast<signed char>(value); break;
            default: break;
        }
    }

    enum class ActivationFunction { IDENTITY, RELU, GELU, TANH, FAST_TANH, SIGMOID };
    enum class LayerType { DENSE, GRU, CONV2D, MAX_POOL2D, AVG_POOL2D, FLATTEN, UNFLATTEN, SAMPLE_AND_SQUASH, STANDARDIZE, EMBEDDING, SEQUENTIAL, PARALLEL, MLP, RESNET_BLOCK };

    template <typename T_TI>
    struct TensorSpecification{
        using TI = T_TI;
        static constexpr TI MAX_RANK = 5;
    };

    template <typename T_SPEC>
    struct Tensor{
        using SPEC = T_SPEC;
        using TI = typename SPEC::TI;
        void* data = nullptr;
        TI shape[SPEC::MAX_RANK] = {};
        TI rank = 0;
        TI size = 0;
        Type type = Type::FLOAT32;
    };

    template <typename T_TI> struct Layer;
    template <typename T_TI> struct State;

    namespace layers{
        template <typename T_TI>
        struct Dense{
            using TI = T_TI;
            using TENSOR_SPEC = TensorSpecification<TI>;
            Tensor<TENSOR_SPEC> weights;
            Tensor<TENSOR_SPEC> biases;
            ActivationFunction activation_function;
            TI input_dim;
            TI output_dim;
        };
        template <typename T_TI>
        struct GRU{
            using TI = T_TI;
            using TENSOR_SPEC = TensorSpecification<TI>;
            Tensor<TENSOR_SPEC> weights_input;
            Tensor<TENSOR_SPEC> biases_input;
            Tensor<TENSOR_SPEC> weights_hidden;
            Tensor<TENSOR_SPEC> biases_hidden;
            Tensor<TENSOR_SPEC> initial_hidden_state;
            TI input_dim;
            TI hidden_dim;
        };
        template <typename T_TI>
        struct Conv2d{
            using TI = T_TI;
            using TENSOR_SPEC = TensorSpecification<TI>;
            Tensor<TENSOR_SPEC> weights;
            Tensor<TENSOR_SPEC> biases;
            ActivationFunction activation_function;
            TI output_channels, input_channels;
            TI kernel_height, kernel_width;
            TI stride_h, stride_w;
            TI padding_h, padding_w;
            enum class Normalization { NONE, BATCH_NORM, LAYER_NORM } normalization = Normalization::NONE;
            Tensor<TENSOR_SPEC> gamma;
            Tensor<TENSOR_SPEC> beta;
            Tensor<TENSOR_SPEC> running_mean;
            Tensor<TENSOR_SPEC> running_var;
        };
        template <typename T_TI>
        struct MaxPool2d{
            using TI = T_TI;
            TI kernel_height, kernel_width;
            TI stride_h, stride_w;
            TI padding_h, padding_w;
        };
        struct AvgPool2d{};
        struct Flatten{};
        struct Unflatten{};
        struct SampleAndSquash{};
        template <typename T_TI>
        struct Standardize{
            using TI = T_TI;
            using TENSOR_SPEC = TensorSpecification<TI>;
            Tensor<TENSOR_SPEC> mean;
            Tensor<TENSOR_SPEC> precision;
            TI dim;
        };
        template <typename T_TI>
        struct Embedding{
            using TI = T_TI;
            using TENSOR_SPEC = TensorSpecification<TI>;
            Tensor<TENSOR_SPEC> weights;
            TI num_classes;
            TI embedding_dim;
        };
        template <typename T_TI>
        struct Sequential{
            using TI = T_TI;
            Layer<TI>* layers = nullptr;
            TI num_layers = 0;
        };
        template <typename T_TI>
        struct MLP{
            using TI = T_TI;
            Layer<TI> input_layer;
            Layer<TI>* hidden_layers = nullptr;
            TI num_hidden_layers = 0;
            Layer<TI> output_layer;
        };
        template <typename T_TI>
        struct Parallel{
            using TI = T_TI;
            Layer<TI>* pipeline_a = nullptr;
            Layer<TI>* pipeline_b = nullptr;
            Layer<TI>* head = nullptr;
        };
        template <typename T_TI>
        struct ResnetBlock{
            using TI = T_TI;
            Layer<TI> conv1;
            Layer<TI> conv2;
            Layer<TI>* downsample = nullptr;
        };
    }

    template <typename T_TI>
    struct Layer{
        using TI = T_TI;
        LayerType type;
        void* data = nullptr;
    };

    template <typename T_TI>
    struct State{
        using TI = T_TI;
        TI batch_size = 0;
        const Layer<TI>* layer = nullptr;
        LayerType type;
        void* data = nullptr;
    };

    namespace state{
        template <typename T_TI>
        struct GRU{
            using TI = T_TI;
            using TENSOR_SPEC = TensorSpecification<TI>;
            Tensor<TENSOR_SPEC> hidden;
            bool initialized = false;
        };
        template <typename T_TI>
        struct Sequential{
            using TI = T_TI;
            State<TI>* layer_states = nullptr;
            TI num_layers = 0;
        };
        template <typename T_TI>
        struct MLP{
            using TI = T_TI;
            State<TI> input_layer_state;
            State<TI>* hidden_layer_states = nullptr;
            TI num_hidden_layers = 0;
            State<TI> output_layer_state;
        };
    }

    template <typename T_TI>
    struct Buffer{
        using TI = T_TI;
        TI batch_size = 0;
        const Layer<TI>* layer = nullptr;
        TI input_shape[TensorSpecification<TI>::MAX_RANK] = {};
        TI input_rank = 0;
        TI input_size = 0;
        Tensor<TensorSpecification<TI>> tick;
        Tensor<TensorSpecification<TI>> tock;
        Tensor<TensorSpecification<TI>> gru_state_scratch;
        Tensor<TensorSpecification<TI>> gru_gate_scratch;
        Tensor<TensorSpecification<TI>> resnet_intermediate;
        Tensor<TensorSpecification<TI>> resnet_shortcut;
        TI max_size = 0;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif

#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_DYNAMIC_CONV2D_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_DYNAMIC_CONV2D_OPERATIONS_GENERIC_H

#include "../../../containers/tensor/tensor.h"
#include "layer.h"
#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    // ======================== malloc / free ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::dynamic_conv2d::LayerForward<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::dynamic_conv2d::LayerForward<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::dynamic_conv2d::LayerBackward<SPEC>& layer) {
        malloc(device, layer.pre_activations);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::dynamic_conv2d::LayerBackward<SPEC>& layer) {
        free(device, layer.pre_activations);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::dynamic_conv2d::LayerGradient<SPEC>& layer) {
        malloc(device, static_cast<nn::layers::dynamic_conv2d::LayerBackward<SPEC>&>(layer));
        malloc(device, layer.output);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::dynamic_conv2d::LayerGradient<SPEC>& layer) {
        free(device, layer.output);
        free(device, static_cast<nn::layers::dynamic_conv2d::LayerBackward<SPEC>&>(layer));
    }
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::dynamic_conv2d::State&) {}
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::dynamic_conv2d::State&) {}
    template<typename DEVICE, typename BUFFER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::dynamic_conv2d::Buffer<BUFFER_SPEC>& buffer) {
        malloc(device, buffer.d_data_acc);
        malloc(device, buffer.d_kernel_weights_acc);
    }
    template<typename DEVICE, typename BUFFER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::dynamic_conv2d::Buffer<BUFFER_SPEC>& buffer) {
        free(device, buffer.d_data_acc);
        free(device, buffer.d_kernel_weights_acc);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE&, TARGET_DEVICE&, nn::layers::dynamic_conv2d::State&, nn::layers::dynamic_conv2d::State&) {}
    template<typename SPEC, typename DEVICE, typename RNG, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE&, const nn::layers::dynamic_conv2d::LayerForward<SPEC>&, nn::layers::dynamic_conv2d::State&, RNG&, Mode<MODE>) {}

    // ======================== init_weights (no-op) ========================
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init_weights(DEVICE& device, nn::layers::dynamic_conv2d::LayerForward<SPEC>& layer, RNG& rng) {}

    // ======================== evaluate ========================
#ifndef RL_TOOLS_NN_DISABLE_GENERIC_FORWARD_BACKWARD
    template<typename DEVICE, typename LAYER_SPEC, typename DATA_SPEC, typename KERNEL_WEIGHTS_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, const nn::layers::dynamic_conv2d::LayerForward<LAYER_SPEC>& layer, const Tensor<DATA_SPEC>& data, const Tensor<KERNEL_WEIGHTS_SPEC>& kernel_weights, Tensor<OUTPUT_SPEC>& output, nn::layers::dynamic_conv2d::Buffer<BUFFER_SPEC>&, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        static_assert(nn::layers::dynamic_conv2d::check_data_output<LAYER_SPEC, DATA_SPEC, OUTPUT_SPEC>);
        using TI = typename DEVICE::index_t;
        using T = typename OUTPUT_SPEC::T;
        using ACCUMULATOR_TYPE = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Accumulator>;
        constexpr TI BATCH_SIZE = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr TI C = LAYER_SPEC::INPUT_CHANNELS;

        using INTERNAL_DATA_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH, C>;
        using INTERNAL_KERNEL_WEIGHTS_SHAPE = tensor::Shape<TI, BATCH_SIZE, C, LAYER_SPEC::KERNEL_HEIGHT, LAYER_SPEC::KERNEL_WIDTH>;
        using INTERNAL_OUTPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::OUTPUT_HEIGHT, LAYER_SPEC::OUTPUT_WIDTH, C>;
        auto data_4d = view_memory<INTERNAL_DATA_SHAPE>(device, data);
        auto kw_4d = view_memory<INTERNAL_KERNEL_WEIGHTS_SHAPE>(device, kernel_weights);
        auto output_4d = view_memory<INTERNAL_OUTPUT_SHAPE>(device, output);

        for(TI bi = 0; bi < BATCH_SIZE; bi++){
            for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                    for(TI c = 0; c < C; c++){
                        ACCUMULATOR_TYPE acc = 0;
                        for(TI kh = 0; kh < LAYER_SPEC::KERNEL_HEIGHT; kh++){
                            for(TI kw = 0; kw < LAYER_SPEC::KERNEL_WIDTH; kw++){
                                TI ih_padded = oh * LAYER_SPEC::STRIDE_H + kh;
                                TI iw_padded = ow * LAYER_SPEC::STRIDE_W + kw;
                                if(ih_padded >= LAYER_SPEC::PADDING_H && ih_padded < LAYER_SPEC::INPUT_HEIGHT + LAYER_SPEC::PADDING_H &&
                                   iw_padded >= LAYER_SPEC::PADDING_W && iw_padded < LAYER_SPEC::INPUT_WIDTH + LAYER_SPEC::PADDING_W){
                                    TI ih = ih_padded - LAYER_SPEC::PADDING_H;
                                    TI iw = iw_padded - LAYER_SPEC::PADDING_W;
                                    acc += (ACCUMULATOR_TYPE)get(device, kw_4d, bi, c, kh, kw) * (ACCUMULATOR_TYPE)get(device, data_4d, bi, ih, iw, c);
                                }
                            }
                        }
                        set(device, output_4d, (T)activation<typename DEVICE::SPEC::MATH, ACCUMULATOR_TYPE, LAYER_SPEC::ACTIVATION_FUNCTION>(acc), bi, oh, ow, c);
                    }
                }
            }
        }
    }

    // ======================== forward (LayerBackward — stores pre-activations) ========================
    template<typename DEVICE, typename LAYER_SPEC, typename DATA_SPEC, typename KERNEL_WEIGHTS_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::dynamic_conv2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<DATA_SPEC>& data, const Tensor<KERNEL_WEIGHTS_SPEC>& kernel_weights, Tensor<OUTPUT_SPEC>& output, nn::layers::dynamic_conv2d::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::dynamic_conv2d::check_data_output<LAYER_SPEC, DATA_SPEC, OUTPUT_SPEC>);
        using TI = typename DEVICE::index_t;
        using T = typename OUTPUT_SPEC::T;
        using ACCUMULATOR_TYPE = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Accumulator>;
        constexpr TI BATCH_SIZE = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr TI C = LAYER_SPEC::INPUT_CHANNELS;

        using INTERNAL_DATA_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH, C>;
        using INTERNAL_KERNEL_WEIGHTS_SHAPE = tensor::Shape<TI, BATCH_SIZE, C, LAYER_SPEC::KERNEL_HEIGHT, LAYER_SPEC::KERNEL_WIDTH>;
        using INTERNAL_OUTPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::OUTPUT_HEIGHT, LAYER_SPEC::OUTPUT_WIDTH, C>;
        auto data_4d = view_memory<INTERNAL_DATA_SHAPE>(device, data);
        auto kw_4d = view_memory<INTERNAL_KERNEL_WEIGHTS_SHAPE>(device, kernel_weights);
        auto output_4d = view_memory<INTERNAL_OUTPUT_SHAPE>(device, output);

        for(TI bi = 0; bi < BATCH_SIZE; bi++){
            for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                    for(TI c = 0; c < C; c++){
                        ACCUMULATOR_TYPE acc = 0;
                        for(TI kh = 0; kh < LAYER_SPEC::KERNEL_HEIGHT; kh++){
                            for(TI kw = 0; kw < LAYER_SPEC::KERNEL_WIDTH; kw++){
                                TI ih_padded = oh * LAYER_SPEC::STRIDE_H + kh;
                                TI iw_padded = ow * LAYER_SPEC::STRIDE_W + kw;
                                if(ih_padded >= LAYER_SPEC::PADDING_H && ih_padded < LAYER_SPEC::INPUT_HEIGHT + LAYER_SPEC::PADDING_H &&
                                   iw_padded >= LAYER_SPEC::PADDING_W && iw_padded < LAYER_SPEC::INPUT_WIDTH + LAYER_SPEC::PADDING_W){
                                    TI ih = ih_padded - LAYER_SPEC::PADDING_H;
                                    TI iw = iw_padded - LAYER_SPEC::PADDING_W;
                                    acc += (ACCUMULATOR_TYPE)get(device, kw_4d, bi, c, kh, kw) * (ACCUMULATOR_TYPE)get(device, data_4d, bi, ih, iw, c);
                                }
                            }
                        }
                        set(device, layer.pre_activations, (T)acc, bi, oh, ow, c);
                        set(device, output_4d, (T)activation<typename DEVICE::SPEC::MATH, ACCUMULATOR_TYPE, LAYER_SPEC::ACTIVATION_FUNCTION>(acc), bi, oh, ow, c);
                    }
                }
            }
        }
    }

    template<typename DEVICE, typename LAYER_SPEC, typename DATA_SPEC, typename KERNEL_WEIGHTS_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::dynamic_conv2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<DATA_SPEC>& data, const Tensor<KERNEL_WEIGHTS_SPEC>& kernel_weights, nn::layers::dynamic_conv2d::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        forward(device, static_cast<nn::layers::dynamic_conv2d::LayerBackward<LAYER_SPEC>&>(layer), data, kernel_weights, layer.output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename DATA_SPEC, typename KERNEL_WEIGHTS_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::dynamic_conv2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<DATA_SPEC>& data, const Tensor<KERNEL_WEIGHTS_SPEC>& kernel_weights, Tensor<OUTPUT_SPEC>& output, nn::layers::dynamic_conv2d::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        forward(device, layer, data, kernel_weights, buffer, rng, mode);
        copy(device, device, layer.output, output);
    }

    // ======================== backward_input (d_data only) ========================
    template<typename DEVICE, typename LAYER_SPEC, typename KERNEL_WEIGHTS_SPEC, typename D_OUTPUT_SPEC, typename D_DATA_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_input(DEVICE& device, const nn::layers::dynamic_conv2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<KERNEL_WEIGHTS_SPEC>& kernel_weights, const Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_DATA_SPEC>& d_data, nn::layers::dynamic_conv2d::Buffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        using TI = typename DEVICE::index_t;
        using T = typename D_DATA_SPEC::T;
        using ACCUMULATOR_TYPE = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Accumulator>;
        constexpr TI BATCH_SIZE = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr TI C = LAYER_SPEC::INPUT_CHANNELS;

        using INTERNAL_KERNEL_WEIGHTS_SHAPE = tensor::Shape<TI, BATCH_SIZE, C, LAYER_SPEC::KERNEL_HEIGHT, LAYER_SPEC::KERNEL_WIDTH>;
        using INTERNAL_D_OUTPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::OUTPUT_HEIGHT, LAYER_SPEC::OUTPUT_WIDTH, C>;
        auto kw_4d = view_memory<INTERNAL_KERNEL_WEIGHTS_SHAPE>(device, kernel_weights);
        auto d_output_4d = view_memory<INTERNAL_D_OUTPUT_SHAPE>(device, d_output);

        set_all(device, buffer.d_data_acc, (ACCUMULATOR_TYPE)0);
        for(TI bi = 0; bi < BATCH_SIZE; bi++){
            for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                    for(TI c = 0; c < C; c++){
                        ACCUMULATOR_TYPE d_pre_act = d_activation_d_x<typename DEVICE::SPEC::MATH, ACCUMULATOR_TYPE, LAYER_SPEC::ACTIVATION_FUNCTION>((ACCUMULATOR_TYPE)get(device, layer.pre_activations, bi, oh, ow, c)) * (ACCUMULATOR_TYPE)get(device, d_output_4d, bi, oh, ow, c);
                        for(TI kh = 0; kh < LAYER_SPEC::KERNEL_HEIGHT; kh++){
                            for(TI kw = 0; kw < LAYER_SPEC::KERNEL_WIDTH; kw++){
                                TI ih_padded = oh * LAYER_SPEC::STRIDE_H + kh;
                                TI iw_padded = ow * LAYER_SPEC::STRIDE_W + kw;
                                if(ih_padded >= LAYER_SPEC::PADDING_H && ih_padded < LAYER_SPEC::INPUT_HEIGHT + LAYER_SPEC::PADDING_H &&
                                   iw_padded >= LAYER_SPEC::PADDING_W && iw_padded < LAYER_SPEC::INPUT_WIDTH + LAYER_SPEC::PADDING_W){
                                    TI ih = ih_padded - LAYER_SPEC::PADDING_H;
                                    TI iw = iw_padded - LAYER_SPEC::PADDING_W;
                                    increment(device, buffer.d_data_acc, (ACCUMULATOR_TYPE)get(device, kw_4d, bi, c, kh, kw) * d_pre_act, bi, ih, iw, c);
                                }
                            }
                        }
                    }
                }
            }
        }
        using INTERNAL_D_DATA_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH, C>;
        auto d_data_4d = view_memory<INTERNAL_D_DATA_SHAPE>(device, d_data);
        for(TI bi = 0; bi < BATCH_SIZE; bi++) for(TI ih = 0; ih < LAYER_SPEC::INPUT_HEIGHT; ih++) for(TI iw = 0; iw < LAYER_SPEC::INPUT_WIDTH; iw++) for(TI c = 0; c < C; c++)
            set(device, d_data_4d, (T)get(device, buffer.d_data_acc, bi, ih, iw, c), bi, ih, iw, c);
    }

    // ======================== backward_kernel_weights (d_kernel_weights only) ========================
    template<typename DEVICE, typename LAYER_SPEC, typename DATA_SPEC, typename D_OUTPUT_SPEC, typename D_KERNEL_WEIGHTS_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_kernel_weights(DEVICE& device, const nn::layers::dynamic_conv2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<DATA_SPEC>& data, const Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_KERNEL_WEIGHTS_SPEC>& d_kernel_weights, nn::layers::dynamic_conv2d::Buffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        using TI = typename DEVICE::index_t;
        using T = typename D_KERNEL_WEIGHTS_SPEC::T;
        using ACCUMULATOR_TYPE = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Accumulator>;
        constexpr TI BATCH_SIZE = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr TI C = LAYER_SPEC::INPUT_CHANNELS;

        using INTERNAL_DATA_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH, C>;
        using INTERNAL_D_OUTPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::OUTPUT_HEIGHT, LAYER_SPEC::OUTPUT_WIDTH, C>;
        auto data_4d = view_memory<INTERNAL_DATA_SHAPE>(device, data);
        auto d_output_4d = view_memory<INTERNAL_D_OUTPUT_SHAPE>(device, d_output);

        set_all(device, buffer.d_kernel_weights_acc, (ACCUMULATOR_TYPE)0);
        for(TI bi = 0; bi < BATCH_SIZE; bi++){
            for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                    for(TI c = 0; c < C; c++){
                        ACCUMULATOR_TYPE d_pre_act = d_activation_d_x<typename DEVICE::SPEC::MATH, ACCUMULATOR_TYPE, LAYER_SPEC::ACTIVATION_FUNCTION>((ACCUMULATOR_TYPE)get(device, layer.pre_activations, bi, oh, ow, c)) * (ACCUMULATOR_TYPE)get(device, d_output_4d, bi, oh, ow, c);
                        for(TI kh = 0; kh < LAYER_SPEC::KERNEL_HEIGHT; kh++){
                            for(TI kw = 0; kw < LAYER_SPEC::KERNEL_WIDTH; kw++){
                                TI ih_padded = oh * LAYER_SPEC::STRIDE_H + kh;
                                TI iw_padded = ow * LAYER_SPEC::STRIDE_W + kw;
                                if(ih_padded >= LAYER_SPEC::PADDING_H && ih_padded < LAYER_SPEC::INPUT_HEIGHT + LAYER_SPEC::PADDING_H &&
                                   iw_padded >= LAYER_SPEC::PADDING_W && iw_padded < LAYER_SPEC::INPUT_WIDTH + LAYER_SPEC::PADDING_W){
                                    TI ih = ih_padded - LAYER_SPEC::PADDING_H;
                                    TI iw = iw_padded - LAYER_SPEC::PADDING_W;
                                    increment(device, buffer.d_kernel_weights_acc, (ACCUMULATOR_TYPE)get(device, data_4d, bi, ih, iw, c) * d_pre_act, bi, c, kh, kw);
                                }
                            }
                        }
                    }
                }
            }
        }
        using INTERNAL_D_KW_SHAPE = tensor::Shape<TI, BATCH_SIZE, C, LAYER_SPEC::KERNEL_HEIGHT, LAYER_SPEC::KERNEL_WIDTH>;
        auto d_kw_4d = view_memory<INTERNAL_D_KW_SHAPE>(device, d_kernel_weights);
        for(TI bi = 0; bi < BATCH_SIZE; bi++) for(TI c = 0; c < C; c++) for(TI kh = 0; kh < LAYER_SPEC::KERNEL_HEIGHT; kh++) for(TI kw = 0; kw < LAYER_SPEC::KERNEL_WIDTH; kw++)
            set(device, d_kw_4d, (T)get(device, buffer.d_kernel_weights_acc, bi, c, kh, kw), bi, c, kh, kw);
    }

    // ======================== backward_full (d_data + d_kernel_weights, fused) ========================
    template<typename DEVICE, typename LAYER_SPEC, typename DATA_SPEC, typename KERNEL_WEIGHTS_SPEC, typename D_OUTPUT_SPEC, typename D_DATA_SPEC, typename D_KERNEL_WEIGHTS_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_full(DEVICE& device, nn::layers::dynamic_conv2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<DATA_SPEC>& data, const Tensor<KERNEL_WEIGHTS_SPEC>& kernel_weights, Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_DATA_SPEC>& d_data, Tensor<D_KERNEL_WEIGHTS_SPEC>& d_kernel_weights, nn::layers::dynamic_conv2d::Buffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        using TI = typename DEVICE::index_t;
        using T_DATA = typename D_DATA_SPEC::T;
        using T_KW = typename D_KERNEL_WEIGHTS_SPEC::T;
        using ACCUMULATOR_TYPE = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Accumulator>;
        constexpr TI BATCH_SIZE = LAYER_SPEC::INTERNAL_BATCH_SIZE;
        constexpr TI C = LAYER_SPEC::INPUT_CHANNELS;

        using INTERNAL_DATA_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH, C>;
        using INTERNAL_KERNEL_WEIGHTS_SHAPE = tensor::Shape<TI, BATCH_SIZE, C, LAYER_SPEC::KERNEL_HEIGHT, LAYER_SPEC::KERNEL_WIDTH>;
        using INTERNAL_D_OUTPUT_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::OUTPUT_HEIGHT, LAYER_SPEC::OUTPUT_WIDTH, C>;
        auto data_4d = view_memory<INTERNAL_DATA_SHAPE>(device, data);
        auto kw_4d = view_memory<INTERNAL_KERNEL_WEIGHTS_SHAPE>(device, kernel_weights);
        auto d_output_4d = view_memory<INTERNAL_D_OUTPUT_SHAPE>(device, d_output);

        set_all(device, buffer.d_data_acc, (ACCUMULATOR_TYPE)0);
        set_all(device, buffer.d_kernel_weights_acc, (ACCUMULATOR_TYPE)0);
        for(TI bi = 0; bi < BATCH_SIZE; bi++){
            for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                    for(TI c = 0; c < C; c++){
                        ACCUMULATOR_TYPE d_pre_act = d_activation_d_x<typename DEVICE::SPEC::MATH, ACCUMULATOR_TYPE, LAYER_SPEC::ACTIVATION_FUNCTION>((ACCUMULATOR_TYPE)get(device, layer.pre_activations, bi, oh, ow, c)) * (ACCUMULATOR_TYPE)get(device, d_output_4d, bi, oh, ow, c);
                        for(TI kh = 0; kh < LAYER_SPEC::KERNEL_HEIGHT; kh++){
                            for(TI kw = 0; kw < LAYER_SPEC::KERNEL_WIDTH; kw++){
                                TI ih_padded = oh * LAYER_SPEC::STRIDE_H + kh;
                                TI iw_padded = ow * LAYER_SPEC::STRIDE_W + kw;
                                if(ih_padded >= LAYER_SPEC::PADDING_H && ih_padded < LAYER_SPEC::INPUT_HEIGHT + LAYER_SPEC::PADDING_H &&
                                   iw_padded >= LAYER_SPEC::PADDING_W && iw_padded < LAYER_SPEC::INPUT_WIDTH + LAYER_SPEC::PADDING_W){
                                    TI ih = ih_padded - LAYER_SPEC::PADDING_H;
                                    TI iw = iw_padded - LAYER_SPEC::PADDING_W;
                                    increment(device, buffer.d_data_acc, (ACCUMULATOR_TYPE)get(device, kw_4d, bi, c, kh, kw) * d_pre_act, bi, ih, iw, c);
                                    increment(device, buffer.d_kernel_weights_acc, (ACCUMULATOR_TYPE)get(device, data_4d, bi, ih, iw, c) * d_pre_act, bi, c, kh, kw);
                                }
                            }
                        }
                    }
                }
            }
        }
        using INTERNAL_D_DATA_SHAPE = tensor::Shape<TI, BATCH_SIZE, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH, C>;
        using INTERNAL_D_KW_SHAPE = tensor::Shape<TI, BATCH_SIZE, C, LAYER_SPEC::KERNEL_HEIGHT, LAYER_SPEC::KERNEL_WIDTH>;
        auto d_data_4d = view_memory<INTERNAL_D_DATA_SHAPE>(device, d_data);
        auto d_kw_4d = view_memory<INTERNAL_D_KW_SHAPE>(device, d_kernel_weights);
        for(TI bi = 0; bi < BATCH_SIZE; bi++) for(TI ih = 0; ih < LAYER_SPEC::INPUT_HEIGHT; ih++) for(TI iw = 0; iw < LAYER_SPEC::INPUT_WIDTH; iw++) for(TI c = 0; c < C; c++)
            set(device, d_data_4d, (T_DATA)get(device, buffer.d_data_acc, bi, ih, iw, c), bi, ih, iw, c);
        for(TI bi = 0; bi < BATCH_SIZE; bi++) for(TI c = 0; c < C; c++) for(TI kh = 0; kh < LAYER_SPEC::KERNEL_HEIGHT; kh++) for(TI kw = 0; kw < LAYER_SPEC::KERNEL_WIDTH; kw++)
            set(device, d_kw_4d, (T_KW)get(device, buffer.d_kernel_weights_acc, bi, c, kh, kw), bi, c, kh, kw);
    }
#endif

    // ======================== zero_gradient / update / _reset_optimizer_state (no-ops) ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_gradient(DEVICE& device, nn::layers::dynamic_conv2d::LayerGradient<SPEC>& layer) {}
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void update(DEVICE& device, nn::layers::dynamic_conv2d::LayerGradient<SPEC>& layer, OPTIMIZER& optimizer) {}
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state(DEVICE& device, nn::layers::dynamic_conv2d::LayerGradient<SPEC>& layer, OPTIMIZER& optimizer) {}

    // ======================== copy ========================
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::layers::dynamic_conv2d::LayerForward<SOURCE_SPEC>& source, nn::layers::dynamic_conv2d::LayerForward<TARGET_SPEC>& target) {
        static_assert(nn::layers::dynamic_conv2d::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::layers::dynamic_conv2d::LayerBackward<SOURCE_SPEC>& source, nn::layers::dynamic_conv2d::LayerBackward<TARGET_SPEC>& target) {
        static_assert(nn::layers::dynamic_conv2d::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>);
        copy(source_device, target_device, source.pre_activations, target.pre_activations);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::layers::dynamic_conv2d::LayerGradient<SOURCE_SPEC>& source, nn::layers::dynamic_conv2d::LayerGradient<TARGET_SPEC>& target) {
        static_assert(nn::layers::dynamic_conv2d::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>);
        copy(source_device, target_device, static_cast<const nn::layers::dynamic_conv2d::LayerBackward<SOURCE_SPEC>&>(source), static_cast<nn::layers::dynamic_conv2d::LayerBackward<TARGET_SPEC>&>(target));
        copy(source_device, target_device, source.output, target.output);
    }

    // ======================== abs_diff ========================
    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT typename S1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::layers::dynamic_conv2d::LayerForward<S1>&, const nn::layers::dynamic_conv2d::LayerForward<S2>&) { return 0; }
    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT typename S1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::layers::dynamic_conv2d::LayerBackward<S1>& l1, const nn::layers::dynamic_conv2d::LayerBackward<S2>& l2) {
        return abs_diff(device, l1.pre_activations, l2.pre_activations);
    }
    template<typename DEVICE, typename S1, typename S2>
    RL_TOOLS_FUNCTION_PLACEMENT typename S1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::layers::dynamic_conv2d::LayerGradient<S1>& l1, const nn::layers::dynamic_conv2d::LayerGradient<S2>& l2) {
        return abs_diff(device, static_cast<const nn::layers::dynamic_conv2d::LayerBackward<S1>&>(l1), static_cast<const nn::layers::dynamic_conv2d::LayerBackward<S2>&>(l2))
             + abs_diff(device, l1.output, l2.output);
    }
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT auto abs_diff(DEVICE& device, const nn::layers::dynamic_conv2d::State&, const nn::layers::dynamic_conv2d::State&) { return 0; }

    // ======================== reset_forward_state ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn::layers::dynamic_conv2d::LayerBackward<SPEC>& l) {
        set_all(device, l.pre_activations, 0);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn::layers::dynamic_conv2d::LayerGradient<SPEC>& l) {
        reset_forward_state(device, static_cast<nn::layers::dynamic_conv2d::LayerBackward<SPEC>&>(l));
        set_all(device, l.output, 0);
    }

    // ======================== is_nan ========================
    template<typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE&, const nn::layers::dynamic_conv2d::LayerForward<SPEC>&, const Mode<MODE>& = Mode<mode::Default<>>{}) { return false; }
    template<typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const nn::layers::dynamic_conv2d::LayerBackward<SPEC>& l, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        return is_nan(device, l.pre_activations, mode);
    }
    template<typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const nn::layers::dynamic_conv2d::LayerGradient<SPEC>& l, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        return is_nan(device, static_cast<const nn::layers::dynamic_conv2d::LayerBackward<SPEC>&>(l), mode) || is_nan(device, l.output, mode);
    }
    template<typename DEVICE, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE&, nn::layers::dynamic_conv2d::State&, const Mode<MODE>& = Mode<mode::Default<>>{}) { return false; }

    // ======================== output ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto output(DEVICE& device, nn::layers::dynamic_conv2d::LayerGradient<SPEC>& l){
        return view_memory<typename SPEC::OUTPUT_SHAPE>(device, l.output);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif

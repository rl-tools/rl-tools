#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_CONV2D_OPERATIONS_CPU_MKL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_CONV2D_OPERATIONS_CPU_MKL_H

#include "../../../devices/cpu_mkl.h"
#include "operations_generic.h"

#ifdef RL_TOOLS_BACKEND_ENABLE_DNNL
#include <dnnl.hpp>
#include <unordered_map>
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
#ifdef RL_TOOLS_BACKEND_ENABLE_DNNL
    namespace nn::layers::conv2d::dnnl_helper{
        template<typename LAYER_SPEC>
        constexpr bool can_use_dnnl =
            LAYER_SPEC::NORMALIZATION == nn::layers::conv2d::Normalization::NONE &&
            utils::typing::is_same_v<typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Parameter>, float>;

        struct DnnlContext{
            dnnl::engine engine;
            dnnl::stream stream;
            DnnlContext(): engine(dnnl::engine::kind::cpu, 0), stream(engine){}
        };

        struct ConvBundle{
            dnnl::primitive prim;
            dnnl::memory src, weights, bias, dst;
            // Optional reorder primitives/memories for when DNNL wants different formats
            dnnl::memory user_weights;
            dnnl::primitive weights_reorder;
            bool needs_weights_reorder = false;
        };

        struct BwdDataBundle{
            dnnl::primitive prim;
            dnnl::memory diff_dst, weights, diff_src;
            dnnl::memory user_weights;
            dnnl::primitive weights_reorder;
            bool needs_weights_reorder = false;
        };

        struct BwdWeightsBundle{
            dnnl::primitive prim;
            dnnl::memory src, diff_dst, diff_weights, diff_bias;
            dnnl::memory user_diff_weights;
            dnnl::primitive diff_weights_reorder;
            bool needs_diff_weights_reorder = false;
        };

        struct DnnlState{
            ConvBundle fwd_train;
            BwdDataBundle bwd_data;
            BwdWeightsBundle bwd_weights;
            std::unordered_map<int64_t, ConvBundle> fwd_infer_cache;
        };

        template<typename DEV_SPEC>
        DnnlContext& get_context(devices::CPU_MKL<DEV_SPEC>& device){
            if(!device.dnnl_context){
                device.dnnl_context = new DnnlContext();
            }
            return *static_cast<DnnlContext*>(device.dnnl_context);
        }

        inline dnnl::memory::desc make_nhwc_md(dnnl::memory::dim n, dnnl::memory::dim c, dnnl::memory::dim h, dnnl::memory::dim w){
            return dnnl::memory::desc({n, c, h, w}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nhwc);
        }

        template<nn::activation_functions::ActivationFunction AF>
        dnnl::post_ops make_activation_post_ops(){
            dnnl::post_ops ops;
            if constexpr(AF == nn::activation_functions::ActivationFunction::RELU){
                ops.append_eltwise(dnnl::algorithm::eltwise_relu, 0.f, 0.f);
            }
            else if constexpr(AF == nn::activation_functions::ActivationFunction::TANH || AF == nn::activation_functions::ActivationFunction::FAST_TANH){
                ops.append_eltwise(dnnl::algorithm::eltwise_tanh, 0.f, 0.f);
            }
            else if constexpr(AF == nn::activation_functions::ActivationFunction::SIGMOID){
                ops.append_eltwise(dnnl::algorithm::eltwise_logistic, 0.f, 0.f);
            }
            return ops;
        }

        template<typename LAYER_SPEC>
        ConvBundle create_fwd_bundle(dnnl::engine& engine, dnnl::memory::dim batch_size, bool fuse_activation){
            auto src_md = make_nhwc_md(batch_size, LAYER_SPEC::INPUT_CHANNELS, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH);
            auto user_weights_md = make_nhwc_md(LAYER_SPEC::OUTPUT_CHANNELS, LAYER_SPEC::INPUT_CHANNELS, LAYER_SPEC::KERNEL_HEIGHT, LAYER_SPEC::KERNEL_WIDTH);
            auto dst_md = make_nhwc_md(batch_size, LAYER_SPEC::OUTPUT_CHANNELS, LAYER_SPEC::OUTPUT_HEIGHT, LAYER_SPEC::OUTPUT_WIDTH);
            auto bias_md = dnnl::memory::desc({(dnnl::memory::dim)LAYER_SPEC::OUTPUT_CHANNELS}, dnnl::memory::data_type::f32, {1});
            // Let DNNL choose optimal weight format
            auto any_weights_md = dnnl::memory::desc({(dnnl::memory::dim)LAYER_SPEC::OUTPUT_CHANNELS, (dnnl::memory::dim)LAYER_SPEC::INPUT_CHANNELS, (dnnl::memory::dim)LAYER_SPEC::KERNEL_HEIGHT, (dnnl::memory::dim)LAYER_SPEC::KERNEL_WIDTH}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);

            dnnl::memory::dims strides = {(dnnl::memory::dim)LAYER_SPEC::STRIDE_H, (dnnl::memory::dim)LAYER_SPEC::STRIDE_W};
            dnnl::memory::dims padding = {(dnnl::memory::dim)LAYER_SPEC::PADDING_H, (dnnl::memory::dim)LAYER_SPEC::PADDING_W};

            auto prop = fuse_activation ? dnnl::prop_kind::forward_inference : dnnl::prop_kind::forward_training;
            dnnl::primitive_attr attr;
            if(fuse_activation && LAYER_SPEC::ACTIVATION_FUNCTION != nn::activation_functions::ActivationFunction::IDENTITY){
                attr.set_post_ops(make_activation_post_ops<LAYER_SPEC::ACTIVATION_FUNCTION>());
            }

            auto pd = dnnl::convolution_forward::primitive_desc(engine, prop, dnnl::algorithm::convolution_auto,
                src_md, any_weights_md, bias_md, dst_md, strides, padding, padding, attr);

            ConvBundle bundle;
            bundle.prim = dnnl::convolution_forward(pd);
            bundle.src = dnnl::memory(src_md, engine);
            bundle.bias = dnnl::memory(bias_md, engine);
            bundle.dst = dnnl::memory(dst_md, engine);

            // Check if DNNL wants a different weight format
            auto chosen_weights_md = pd.weights_desc();
            if(chosen_weights_md != user_weights_md){
                bundle.user_weights = dnnl::memory(user_weights_md, engine);
                bundle.weights = dnnl::memory(chosen_weights_md, engine);
                bundle.weights_reorder = dnnl::reorder(bundle.user_weights, bundle.weights);
                bundle.needs_weights_reorder = true;
            }
            else{
                bundle.weights = dnnl::memory(user_weights_md, engine);
            }
            return bundle;
        }

        template<typename LAYER_SPEC>
        void create_training_bundles(dnnl::engine& engine, DnnlState& state){
            constexpr dnnl::memory::dim BS = LAYER_SPEC::INTERNAL_BATCH_SIZE;
            auto src_md = make_nhwc_md(BS, LAYER_SPEC::INPUT_CHANNELS, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH);
            auto user_weights_md = make_nhwc_md(LAYER_SPEC::OUTPUT_CHANNELS, LAYER_SPEC::INPUT_CHANNELS, LAYER_SPEC::KERNEL_HEIGHT, LAYER_SPEC::KERNEL_WIDTH);
            auto any_weights_md = dnnl::memory::desc({(dnnl::memory::dim)LAYER_SPEC::OUTPUT_CHANNELS, (dnnl::memory::dim)LAYER_SPEC::INPUT_CHANNELS, (dnnl::memory::dim)LAYER_SPEC::KERNEL_HEIGHT, (dnnl::memory::dim)LAYER_SPEC::KERNEL_WIDTH}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
            auto dst_md = make_nhwc_md(BS, LAYER_SPEC::OUTPUT_CHANNELS, LAYER_SPEC::OUTPUT_HEIGHT, LAYER_SPEC::OUTPUT_WIDTH);
            auto bias_md = dnnl::memory::desc({(dnnl::memory::dim)LAYER_SPEC::OUTPUT_CHANNELS}, dnnl::memory::data_type::f32, {1});

            dnnl::memory::dims strides = {(dnnl::memory::dim)LAYER_SPEC::STRIDE_H, (dnnl::memory::dim)LAYER_SPEC::STRIDE_W};
            dnnl::memory::dims padding = {(dnnl::memory::dim)LAYER_SPEC::PADDING_H, (dnnl::memory::dim)LAYER_SPEC::PADDING_W};

            // Forward training (no activation fusion — need pre_activations for backward)
            state.fwd_train = create_fwd_bundle<LAYER_SPEC>(engine, BS, false);

            // Backward data
            {
                auto fwd_pd = dnnl::convolution_forward::primitive_desc(engine, dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_auto,
                    src_md, any_weights_md, bias_md, dst_md, strides, padding, padding);
                auto pd = dnnl::convolution_backward_data::primitive_desc(engine, dnnl::algorithm::convolution_auto,
                    src_md, any_weights_md, dst_md, strides, padding, padding, fwd_pd);
                auto chosen_weights_md = pd.weights_desc();
                state.bwd_data.prim = dnnl::convolution_backward_data(pd);
                state.bwd_data.diff_dst = dnnl::memory(dst_md, engine);
                state.bwd_data.diff_src = dnnl::memory(src_md, engine);
                if(chosen_weights_md != user_weights_md){
                    state.bwd_data.user_weights = dnnl::memory(user_weights_md, engine);
                    state.bwd_data.weights = dnnl::memory(chosen_weights_md, engine);
                    state.bwd_data.weights_reorder = dnnl::reorder(state.bwd_data.user_weights, state.bwd_data.weights);
                    state.bwd_data.needs_weights_reorder = true;
                }
                else{
                    state.bwd_data.weights = dnnl::memory(user_weights_md, engine);
                }
            }

            // Backward weights
            {
                auto fwd_pd = dnnl::convolution_forward::primitive_desc(engine, dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_auto,
                    src_md, any_weights_md, bias_md, dst_md, strides, padding, padding);
                auto pd = dnnl::convolution_backward_weights::primitive_desc(engine, dnnl::algorithm::convolution_auto,
                    src_md, any_weights_md, bias_md, dst_md, strides, padding, padding, fwd_pd);
                auto chosen_diff_weights_md = pd.diff_weights_desc();
                state.bwd_weights.prim = dnnl::convolution_backward_weights(pd);
                state.bwd_weights.src = dnnl::memory(src_md, engine);
                state.bwd_weights.diff_dst = dnnl::memory(dst_md, engine);
                state.bwd_weights.diff_bias = dnnl::memory(bias_md, engine);
                if(chosen_diff_weights_md != user_weights_md){
                    state.bwd_weights.diff_weights = dnnl::memory(chosen_diff_weights_md, engine);
                    state.bwd_weights.user_diff_weights = dnnl::memory(user_weights_md, engine);
                    state.bwd_weights.diff_weights_reorder = dnnl::reorder(state.bwd_weights.diff_weights, state.bwd_weights.user_diff_weights);
                    state.bwd_weights.needs_diff_weights_reorder = true;
                }
                else{
                    state.bwd_weights.diff_weights = dnnl::memory(user_weights_md, engine);
                }
            }
        }

        inline void exec_fwd(ConvBundle& bundle, dnnl::stream& stream, void* src_ptr, void* weights_ptr, void* bias_ptr, void* dst_ptr){
            bundle.src.set_data_handle(src_ptr);
            bundle.bias.set_data_handle(bias_ptr);
            bundle.dst.set_data_handle(dst_ptr);
            if(bundle.needs_weights_reorder){
                bundle.user_weights.set_data_handle(weights_ptr);
                bundle.weights_reorder.execute(stream, {{DNNL_ARG_FROM, bundle.user_weights}, {DNNL_ARG_TO, bundle.weights}});
            }
            else{
                bundle.weights.set_data_handle(weights_ptr);
            }
            bundle.prim.execute(stream, {
                {DNNL_ARG_SRC, bundle.src}, {DNNL_ARG_WEIGHTS, bundle.weights},
                {DNNL_ARG_BIAS, bundle.bias}, {DNNL_ARG_DST, bundle.dst}});
            stream.wait();
        }

        template<typename DEVICE, typename LAYER_SPEC, typename TENSOR_SPEC>
        void apply_d_activation_elementwise(DEVICE& device, const nn::layers::conv2d::LayerBackward<LAYER_SPEC>& layer, Tensor<TENSOR_SPEC>& d_output){
            using TI = typename DEVICE::index_t;
            using T = typename TENSOR_SPEC::T;
            using ACCUMULATOR_TYPE = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Accumulator>;
            constexpr TI BS = LAYER_SPEC::INTERNAL_BATCH_SIZE;
            using SHAPE_4D = tensor::Shape<TI, BS, LAYER_SPEC::OUTPUT_HEIGHT, LAYER_SPEC::OUTPUT_WIDTH, LAYER_SPEC::OUTPUT_CHANNELS>;
            auto d_output_view = view_memory<SHAPE_4D>(device, d_output);
            for(TI bi = 0; bi < BS; bi++){
                for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                    for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                        for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                            ACCUMULATOR_TYPE d_act = d_activation_d_x<typename DEVICE::SPEC::MATH, ACCUMULATOR_TYPE, LAYER_SPEC::ACTIVATION_FUNCTION>((ACCUMULATOR_TYPE)get(device, layer.pre_activations, bi, oh, ow, oc));
                            ACCUMULATOR_TYPE d_out = (ACCUMULATOR_TYPE)get(device, d_output_view, bi, oh, ow, oc);
                            set(device, d_output_view, (T)(d_out * d_act), bi, oh, ow, oc);
                        }
                    }
                }
            }
        }

        template<typename TI, typename T>
        void accumulate(T* dst, const T* src, TI count){
            for(TI i = 0; i < count; i++){
                dst[i] += src[i];
            }
        }

        template<typename BUFFER_SPEC>
        DnnlState& get_state(nn::layers::conv2d::Buffer<BUFFER_SPEC>& buffer){
            return *static_cast<DnnlState*>(buffer.backend_state);
        }
    }

    // ======================== malloc / free overloads for CPU_MKL ========================
    template<typename DEV_SPEC, typename BUFFER_SPEC>
    void malloc(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::conv2d::Buffer<BUFFER_SPEC>& buffer){
        using LAYER_SPEC = typename BUFFER_SPEC::SPEC;
        malloc(static_cast<devices::CPU_BLAS<DEV_SPEC>&>(device), buffer);
        if constexpr(nn::layers::conv2d::dnnl_helper::can_use_dnnl<LAYER_SPEC>){
            auto& ctx = nn::layers::conv2d::dnnl_helper::get_context(device);
            auto* state = new nn::layers::conv2d::dnnl_helper::DnnlState();
            nn::layers::conv2d::dnnl_helper::create_training_bundles<LAYER_SPEC>(ctx.engine, *state);
            buffer.backend_state = state;
        }
    }

    template<typename DEV_SPEC, typename BUFFER_SPEC>
    void free(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::conv2d::Buffer<BUFFER_SPEC>& buffer){
        if(buffer.backend_state){
            delete static_cast<nn::layers::conv2d::dnnl_helper::DnnlState*>(buffer.backend_state);
            buffer.backend_state = nullptr;
        }
        free(static_cast<devices::CPU_BLAS<DEV_SPEC>&>(device), buffer);
    }

    // ======================== evaluate (inference, fused activation) ========================
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE>
    void evaluate(devices::CPU_MKL<DEV_SPEC>& device, const nn::layers::conv2d::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::conv2d::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode){
        static_assert(nn::layers::conv2d::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using TI = typename devices::CPU_MKL<DEV_SPEC>::index_t;
        constexpr TI ACTUAL_BATCH_SIZE = product(typename INPUT_SPEC::SHAPE{}) / (LAYER_SPEC::INPUT_HEIGHT * LAYER_SPEC::INPUT_WIDTH * LAYER_SPEC::INPUT_CHANNELS);
        if constexpr(!nn::layers::conv2d::dnnl_helper::can_use_dnnl<LAYER_SPEC>){
            evaluate(static_cast<devices::CPU_BLAS<DEV_SPEC>&>(device), layer, input, output, buffer, rng, mode);
            return;
        }
        else{
            using INPUT_4D = tensor::Shape<TI, ACTUAL_BATCH_SIZE, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH, LAYER_SPEC::INPUT_CHANNELS>;
            using OUTPUT_4D = tensor::Shape<TI, ACTUAL_BATCH_SIZE, LAYER_SPEC::OUTPUT_HEIGHT, LAYER_SPEC::OUTPUT_WIDTH, LAYER_SPEC::OUTPUT_CHANNELS>;
            auto input_view = view_memory<INPUT_4D>(device, input);
            auto output_view = view_memory<OUTPUT_4D>(device, output);

            auto& ctx = nn::layers::conv2d::dnnl_helper::get_context(device);
            auto& state = nn::layers::conv2d::dnnl_helper::get_state(buffer);

            // Use cached training bundle if batch size matches, otherwise get/create from infer cache
            nn::layers::conv2d::dnnl_helper::ConvBundle* bundle;
            if constexpr(ACTUAL_BATCH_SIZE == LAYER_SPEC::INTERNAL_BATCH_SIZE){
                if constexpr(LAYER_SPEC::ACTIVATION_FUNCTION == nn::activation_functions::ActivationFunction::IDENTITY){
                    bundle = &state.fwd_train;
                }
                else{
                    auto it = state.fwd_infer_cache.find(ACTUAL_BATCH_SIZE);
                    if(it == state.fwd_infer_cache.end()){
                        auto [inserted, _] = state.fwd_infer_cache.emplace(ACTUAL_BATCH_SIZE,
                            nn::layers::conv2d::dnnl_helper::create_fwd_bundle<LAYER_SPEC>(ctx.engine, ACTUAL_BATCH_SIZE, true));
                        bundle = &inserted->second;
                    }
                    else{
                        bundle = &it->second;
                    }
                }
            }
            else{
                auto it = state.fwd_infer_cache.find(ACTUAL_BATCH_SIZE);
                if(it == state.fwd_infer_cache.end()){
                    auto [inserted, _] = state.fwd_infer_cache.emplace(ACTUAL_BATCH_SIZE,
                        nn::layers::conv2d::dnnl_helper::create_fwd_bundle<LAYER_SPEC>(ctx.engine, ACTUAL_BATCH_SIZE, true));
                    bundle = &inserted->second;
                }
                else{
                    bundle = &it->second;
                }
            }

            nn::layers::conv2d::dnnl_helper::exec_fwd(*bundle, ctx.stream, (void*)data(input_view), (void*)data(layer.weights.parameters), (void*)data(layer.biases.parameters), (void*)data(output_view));
        }
    }

    // ======================== evaluate_step ========================
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE>
    void evaluate_step(devices::CPU_MKL<DEV_SPEC>& device, const nn::layers::conv2d::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, nn::layers::conv2d::State& state, Tensor<OUTPUT_SPEC>& output, nn::layers::conv2d::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode){
        evaluate(device, layer, input, output, buffer, rng, mode);
    }

    // ======================== forward (LayerBackward, training, no activation fusion) ========================
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE>
    void forward(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::conv2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::conv2d::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode){
        static_assert(nn::layers::conv2d::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        if constexpr(!nn::layers::conv2d::dnnl_helper::can_use_dnnl<LAYER_SPEC>){
            forward(static_cast<devices::CPU_BLAS<DEV_SPEC>&>(device), layer, input, output, buffer, rng, mode);
            return;
        }
        else{
            using TI = typename devices::CPU_MKL<DEV_SPEC>::index_t;
            constexpr TI BS = LAYER_SPEC::INTERNAL_BATCH_SIZE;
            using INPUT_4D = tensor::Shape<TI, BS, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH, LAYER_SPEC::INPUT_CHANNELS>;
            auto input_view = view_memory<INPUT_4D>(device, input);

            auto& ctx = nn::layers::conv2d::dnnl_helper::get_context(device);
            auto& state = nn::layers::conv2d::dnnl_helper::get_state(buffer);

            nn::layers::conv2d::dnnl_helper::exec_fwd(state.fwd_train, ctx.stream, (void*)data(input_view), (void*)data(layer.weights.parameters), (void*)data(layer.biases.parameters), (void*)data(layer.pre_activations));

            copy(device, device, layer.pre_activations, output);
            if constexpr(LAYER_SPEC::ACTIVATION_FUNCTION != nn::activation_functions::ActivationFunction::IDENTITY){
                using T = typename OUTPUT_SPEC::T;
                using ACCUMULATOR_TYPE = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Accumulator>;
                using OUTPUT_4D = tensor::Shape<TI, BS, LAYER_SPEC::OUTPUT_HEIGHT, LAYER_SPEC::OUTPUT_WIDTH, LAYER_SPEC::OUTPUT_CHANNELS>;
                auto output_view = view_memory<OUTPUT_4D>(device, output);
                for(TI bi = 0; bi < BS; bi++){
                    for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                        for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                            for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                                ACCUMULATOR_TYPE val = (ACCUMULATOR_TYPE)get(device, output_view, bi, oh, ow, oc);
                                set(device, output_view, (T)activation<typename devices::CPU_MKL<DEV_SPEC>::SPEC::MATH, ACCUMULATOR_TYPE, LAYER_SPEC::ACTIVATION_FUNCTION>(val), bi, oh, ow, oc);
                            }
                        }
                    }
                }
            }
        }
    }

    // ======================== forward (LayerGradient wrappers) ========================
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE>
    void forward(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::conv2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, nn::layers::conv2d::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode){
        forward(device, static_cast<nn::layers::conv2d::LayerBackward<LAYER_SPEC>&>(layer), input, layer.output, buffer, rng, mode);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE>
    void forward(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::conv2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::conv2d::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode){
        static_assert(nn::layers::conv2d::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        forward(device, layer, input, buffer, rng, mode);
        copy(device, device, layer.output, output);
    }

    // ======================== backward_input ========================
    template<typename DEV_SPEC, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE>
    void backward_input(devices::CPU_MKL<DEV_SPEC>& device, const nn::layers::conv2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::conv2d::Buffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode){
        if constexpr(!nn::layers::conv2d::dnnl_helper::can_use_dnnl<LAYER_SPEC>){
            backward_input(static_cast<devices::CPU_BLAS<DEV_SPEC>&>(device), layer, d_output, d_input, buffer, mode);
            return;
        }
        else{
            using TI = typename devices::CPU_MKL<DEV_SPEC>::index_t;
            using T = typename D_OUTPUT_SPEC::T;
            using ACCUMULATOR_TYPE = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Accumulator>;
            constexpr TI BS = LAYER_SPEC::INTERNAL_BATCH_SIZE;
            constexpr TI TOTAL_OUTPUT = BS * LAYER_SPEC::OUTPUT_HEIGHT * LAYER_SPEC::OUTPUT_WIDTH * LAYER_SPEC::OUTPUT_CHANNELS;
            using D_OUTPUT_4D = tensor::Shape<TI, BS, LAYER_SPEC::OUTPUT_HEIGHT, LAYER_SPEC::OUTPUT_WIDTH, LAYER_SPEC::OUTPUT_CHANNELS>;
            using D_INPUT_4D = tensor::Shape<TI, BS, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH, LAYER_SPEC::INPUT_CHANNELS>;
            auto d_output_view = view_memory<D_OUTPUT_4D>(device, d_output);
            auto d_input_view = view_memory<D_INPUT_4D>(device, d_input);

            auto& ctx = nn::layers::conv2d::dnnl_helper::get_context(device);
            auto& state = nn::layers::conv2d::dnnl_helper::get_state(buffer);

            void* diff_dst_ptr;
            T* d_pre_act = nullptr;
            if constexpr(LAYER_SPEC::ACTIVATION_FUNCTION != nn::activation_functions::ActivationFunction::IDENTITY){
                d_pre_act = new T[TOTAL_OUTPUT];
                for(TI bi = 0; bi < BS; bi++){
                    for(TI oh = 0; oh < LAYER_SPEC::OUTPUT_HEIGHT; oh++){
                        for(TI ow = 0; ow < LAYER_SPEC::OUTPUT_WIDTH; ow++){
                            for(TI oc = 0; oc < LAYER_SPEC::OUTPUT_CHANNELS; oc++){
                                ACCUMULATOR_TYPE d_act = d_activation_d_x<typename devices::CPU_MKL<DEV_SPEC>::SPEC::MATH, ACCUMULATOR_TYPE, LAYER_SPEC::ACTIVATION_FUNCTION>((ACCUMULATOR_TYPE)get(device, layer.pre_activations, bi, oh, ow, oc));
                                TI flat_idx = ((bi * LAYER_SPEC::OUTPUT_HEIGHT + oh) * LAYER_SPEC::OUTPUT_WIDTH + ow) * LAYER_SPEC::OUTPUT_CHANNELS + oc;
                                d_pre_act[flat_idx] = (T)((ACCUMULATOR_TYPE)get(device, d_output_view, bi, oh, ow, oc) * d_act);
                            }
                        }
                    }
                }
                diff_dst_ptr = (void*)d_pre_act;
            }
            else{
                diff_dst_ptr = (void*)data(d_output_view);
            }

            auto& b = state.bwd_data;
            b.diff_dst.set_data_handle(diff_dst_ptr);
            b.diff_src.set_data_handle((void*)data(d_input_view));
            if(b.needs_weights_reorder){
                b.user_weights.set_data_handle((void*)data(layer.weights.parameters));
                b.weights_reorder.execute(ctx.stream, {{DNNL_ARG_FROM, b.user_weights}, {DNNL_ARG_TO, b.weights}});
            }
            else{
                b.weights.set_data_handle((void*)data(layer.weights.parameters));
            }
            b.prim.execute(ctx.stream, {
                {DNNL_ARG_DIFF_DST, b.diff_dst}, {DNNL_ARG_WEIGHTS, b.weights},
                {DNNL_ARG_DIFF_SRC, b.diff_src}});
            ctx.stream.wait();

            delete[] d_pre_act;
        }
    }

    // ======================== backward (gradient accumulation only) ========================
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename BUFFER_SPEC, typename MODE>
    void backward(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::conv2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, nn::layers::conv2d::Buffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode){
        if constexpr(!nn::layers::conv2d::dnnl_helper::can_use_dnnl<LAYER_SPEC>){
            backward(static_cast<devices::CPU_BLAS<DEV_SPEC>&>(device), layer, input, d_output, buffer, mode);
            return;
        }
        else{
            using TI = typename devices::CPU_MKL<DEV_SPEC>::index_t;
            constexpr TI BS = LAYER_SPEC::INTERNAL_BATCH_SIZE;
            constexpr TI NUM_WEIGHTS = LAYER_SPEC::OUTPUT_CHANNELS * LAYER_SPEC::KERNEL_HEIGHT * LAYER_SPEC::KERNEL_WIDTH * LAYER_SPEC::INPUT_CHANNELS;
            using INPUT_4D = tensor::Shape<TI, BS, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH, LAYER_SPEC::INPUT_CHANNELS>;
            auto input_view = view_memory<INPUT_4D>(device, input);

            if constexpr(LAYER_SPEC::ACTIVATION_FUNCTION != nn::activation_functions::ActivationFunction::IDENTITY){
                nn::layers::conv2d::dnnl_helper::apply_d_activation_elementwise(device, layer, d_output);
            }

            using D_OUTPUT_4D = tensor::Shape<TI, BS, LAYER_SPEC::OUTPUT_HEIGHT, LAYER_SPEC::OUTPUT_WIDTH, LAYER_SPEC::OUTPUT_CHANNELS>;
            auto d_output_view = view_memory<D_OUTPUT_4D>(device, d_output);

            auto& ctx = nn::layers::conv2d::dnnl_helper::get_context(device);
            auto& state = nn::layers::conv2d::dnnl_helper::get_state(buffer);

            auto& b = state.bwd_weights;
            b.src.set_data_handle((void*)data(input_view));
            b.diff_dst.set_data_handle((void*)data(d_output_view));
            b.diff_bias.set_data_handle((void*)data(buffer.d_biases_acc));
            if(b.needs_diff_weights_reorder){
                // DNNL computes in its preferred format, then we reorder back to user format
                b.prim.execute(ctx.stream, {
                    {DNNL_ARG_SRC, b.src}, {DNNL_ARG_DIFF_DST, b.diff_dst},
                    {DNNL_ARG_DIFF_WEIGHTS, b.diff_weights}, {DNNL_ARG_DIFF_BIAS, b.diff_bias}});
                b.user_diff_weights.set_data_handle((void*)data(buffer.d_weights_acc));
                b.diff_weights_reorder.execute(ctx.stream, {{DNNL_ARG_FROM, b.diff_weights}, {DNNL_ARG_TO, b.user_diff_weights}});
            }
            else{
                b.diff_weights.set_data_handle((void*)data(buffer.d_weights_acc));
                b.prim.execute(ctx.stream, {
                    {DNNL_ARG_SRC, b.src}, {DNNL_ARG_DIFF_DST, b.diff_dst},
                    {DNNL_ARG_DIFF_WEIGHTS, b.diff_weights}, {DNNL_ARG_DIFF_BIAS, b.diff_bias}});
            }
            ctx.stream.wait();

            nn::layers::conv2d::dnnl_helper::accumulate<TI>(data(layer.weights.gradient), data(buffer.d_weights_acc), NUM_WEIGHTS);
            nn::layers::conv2d::dnnl_helper::accumulate<TI>(data(layer.biases.gradient), data(buffer.d_biases_acc), (TI)LAYER_SPEC::OUTPUT_CHANNELS);
        }
    }

    // ======================== backward_full ========================
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE>
    void backward_full(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::conv2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::conv2d::Buffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode){
        if constexpr(!nn::layers::conv2d::dnnl_helper::can_use_dnnl<LAYER_SPEC>){
            backward_full(static_cast<devices::CPU_BLAS<DEV_SPEC>&>(device), layer, input, d_output, d_input, buffer, mode);
            return;
        }
        else{
            using TI = typename devices::CPU_MKL<DEV_SPEC>::index_t;
            constexpr TI BS = LAYER_SPEC::INTERNAL_BATCH_SIZE;
            constexpr TI NUM_WEIGHTS = LAYER_SPEC::OUTPUT_CHANNELS * LAYER_SPEC::KERNEL_HEIGHT * LAYER_SPEC::KERNEL_WIDTH * LAYER_SPEC::INPUT_CHANNELS;
            using INPUT_4D = tensor::Shape<TI, BS, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH, LAYER_SPEC::INPUT_CHANNELS>;
            using D_OUTPUT_4D = tensor::Shape<TI, BS, LAYER_SPEC::OUTPUT_HEIGHT, LAYER_SPEC::OUTPUT_WIDTH, LAYER_SPEC::OUTPUT_CHANNELS>;
            using D_INPUT_4D = tensor::Shape<TI, BS, LAYER_SPEC::INPUT_HEIGHT, LAYER_SPEC::INPUT_WIDTH, LAYER_SPEC::INPUT_CHANNELS>;
            auto input_view = view_memory<INPUT_4D>(device, input);
            auto d_input_view = view_memory<D_INPUT_4D>(device, d_input);

            if constexpr(LAYER_SPEC::ACTIVATION_FUNCTION != nn::activation_functions::ActivationFunction::IDENTITY){
                nn::layers::conv2d::dnnl_helper::apply_d_activation_elementwise(device, layer, d_output);
            }

            auto d_output_view = view_memory<D_OUTPUT_4D>(device, d_output);

            auto& ctx = nn::layers::conv2d::dnnl_helper::get_context(device);
            auto& state = nn::layers::conv2d::dnnl_helper::get_state(buffer);

            // backward data
            {
                auto& b = state.bwd_data;
                b.diff_dst.set_data_handle((void*)data(d_output_view));
                b.diff_src.set_data_handle((void*)data(d_input_view));
                if(b.needs_weights_reorder){
                    b.user_weights.set_data_handle((void*)data(layer.weights.parameters));
                    b.weights_reorder.execute(ctx.stream, {{DNNL_ARG_FROM, b.user_weights}, {DNNL_ARG_TO, b.weights}});
                }
                else{
                    b.weights.set_data_handle((void*)data(layer.weights.parameters));
                }
                b.prim.execute(ctx.stream, {
                    {DNNL_ARG_DIFF_DST, b.diff_dst}, {DNNL_ARG_WEIGHTS, b.weights},
                    {DNNL_ARG_DIFF_SRC, b.diff_src}});
                ctx.stream.wait();
            }

            // backward weights
            {
                auto& b = state.bwd_weights;
                b.src.set_data_handle((void*)data(input_view));
                b.diff_dst.set_data_handle((void*)data(d_output_view));
                b.diff_bias.set_data_handle((void*)data(buffer.d_biases_acc));
                if(b.needs_diff_weights_reorder){
                    b.prim.execute(ctx.stream, {
                        {DNNL_ARG_SRC, b.src}, {DNNL_ARG_DIFF_DST, b.diff_dst},
                        {DNNL_ARG_DIFF_WEIGHTS, b.diff_weights}, {DNNL_ARG_DIFF_BIAS, b.diff_bias}});
                    b.user_diff_weights.set_data_handle((void*)data(buffer.d_weights_acc));
                    b.diff_weights_reorder.execute(ctx.stream, {{DNNL_ARG_FROM, b.diff_weights}, {DNNL_ARG_TO, b.user_diff_weights}});
                }
                else{
                    b.diff_weights.set_data_handle((void*)data(buffer.d_weights_acc));
                    b.prim.execute(ctx.stream, {
                        {DNNL_ARG_SRC, b.src}, {DNNL_ARG_DIFF_DST, b.diff_dst},
                        {DNNL_ARG_DIFF_WEIGHTS, b.diff_weights}, {DNNL_ARG_DIFF_BIAS, b.diff_bias}});
                }
                ctx.stream.wait();
            }

            nn::layers::conv2d::dnnl_helper::accumulate<TI>(data(layer.weights.gradient), data(buffer.d_weights_acc), NUM_WEIGHTS);
            nn::layers::conv2d::dnnl_helper::accumulate<TI>(data(layer.biases.gradient), data(buffer.d_biases_acc), (TI)LAYER_SPEC::OUTPUT_CHANNELS);
        }
    }
#else
    // No DNNL: fall through to generic via CPU_BLAS
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE>
    void evaluate(devices::CPU_MKL<DEV_SPEC>& device, const nn::layers::conv2d::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::conv2d::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode){
        evaluate(static_cast<devices::CPU_BLAS<DEV_SPEC>&>(device), layer, input, output, buffer, rng, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE>
    void evaluate_step(devices::CPU_MKL<DEV_SPEC>& device, const nn::layers::conv2d::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, nn::layers::conv2d::State& state, Tensor<OUTPUT_SPEC>& output, nn::layers::conv2d::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode){
        evaluate(device, layer, input, output, buffer, rng, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE>
    void forward(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::conv2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::conv2d::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode){
        forward(static_cast<devices::CPU_BLAS<DEV_SPEC>&>(device), layer, input, output, buffer, rng, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE>
    void forward(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::conv2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, nn::layers::conv2d::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode){
        forward(static_cast<devices::CPU_BLAS<DEV_SPEC>&>(device), layer, input, buffer, rng, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE>
    void forward(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::conv2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::conv2d::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode){
        forward(static_cast<devices::CPU_BLAS<DEV_SPEC>&>(device), layer, input, output, buffer, rng, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE>
    void backward_input(devices::CPU_MKL<DEV_SPEC>& device, const nn::layers::conv2d::LayerBackward<LAYER_SPEC>& layer, const Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::conv2d::Buffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode){
        backward_input(static_cast<devices::CPU_BLAS<DEV_SPEC>&>(device), layer, d_output, d_input, buffer, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename BUFFER_SPEC, typename MODE>
    void backward(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::conv2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, nn::layers::conv2d::Buffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode){
        backward(static_cast<devices::CPU_BLAS<DEV_SPEC>&>(device), layer, input, d_output, buffer, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE>
    void backward_full(devices::CPU_MKL<DEV_SPEC>& device, nn::layers::conv2d::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::conv2d::Buffer<BUFFER_SPEC>& buffer, const Mode<MODE>& mode){
        backward_full(static_cast<devices::CPU_BLAS<DEV_SPEC>&>(device), layer, input, d_output, d_input, buffer, mode);
    }
#endif
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif

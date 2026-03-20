#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_PARALLEL_MODEL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_PARALLEL_MODEL_H

#include "../sequential/model.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn_models::parallel{

    namespace detail{
        template <typename SHAPE_A, typename SHAPE_B, auto INDEX = 0>
        constexpr bool leading_dims_match(){
            constexpr auto RANK = length(SHAPE_A{});
            static_assert(RANK == length(SHAPE_B{}), "Shapes must have same rank");
            if constexpr(INDEX + 1 >= RANK){
                return true;
            }
            else{
                return get<INDEX>(SHAPE_A{}) == get<INDEX>(SHAPE_B{}) && leading_dims_match<SHAPE_A, SHAPE_B, INDEX + 1>();
            }
        }

        // Empty struct used as placeholder when HEAD is void
        struct Empty{};
    }

    template <typename T_CAPABILITY, typename T_MODULE_A, typename T_MODULE_B, typename T_INPUT_SHAPE_A, typename T_INPUT_SHAPE_B, typename T_HEAD = void>
    struct Specification{
        using CAPABILITY = T_CAPABILITY;
        using MODULE_A = T_MODULE_A;
        using MODULE_B = T_MODULE_B;
        using INPUT_SHAPE_A = T_INPUT_SHAPE_A;
        using INPUT_SHAPE_B = T_INPUT_SHAPE_B;
        using HEAD_MODULE = T_HEAD;

        using PIPELINE_TYPE_A = typename MODULE_A::template Layer<CAPABILITY, INPUT_SHAPE_A>;
        using PIPELINE_TYPE_B = typename MODULE_B::template Layer<CAPABILITY, INPUT_SHAPE_B>;

        using OUTPUT_SHAPE_A = typename PIPELINE_TYPE_A::OUTPUT_SHAPE;
        using OUTPUT_SHAPE_B = typename PIPELINE_TYPE_B::OUTPUT_SHAPE;

        static_assert(length(OUTPUT_SHAPE_A{}) == length(OUTPUT_SHAPE_B{}), "Pipeline output shapes must have same rank");
        static_assert(detail::leading_dims_match<OUTPUT_SHAPE_A, OUTPUT_SHAPE_B>(), "Pipeline output shapes must have matching leading dimensions");

        using TI = typename INPUT_SHAPE_A::TI;
        static constexpr TI LAST_DIM_A = get_last(OUTPUT_SHAPE_A{});
        static constexpr TI LAST_DIM_B = get_last(OUTPUT_SHAPE_B{});
        static constexpr TI LAST_DIM = LAST_DIM_A + LAST_DIM_B;
        static constexpr auto RANK = length(OUTPUT_SHAPE_A{});
        using CONCAT_OUTPUT_SHAPE = tensor::Replace<OUTPUT_SHAPE_A, LAST_DIM, RANK - 1>;

        // HEAD (optional post-processing after concatenation)
        static constexpr bool HAS_HEAD = !utils::typing::is_same_v<HEAD_MODULE, void>;
    private:
        template <bool ENABLED, typename = void>
        struct HeadResolver{
            using HEAD_TYPE = detail::Empty;
            using OUTPUT_SHAPE = CONCAT_OUTPUT_SHAPE;
        };
        template <typename DUMMY>
        struct HeadResolver<true, DUMMY>{
            using HEAD_TYPE = typename HEAD_MODULE::template Layer<CAPABILITY, CONCAT_OUTPUT_SHAPE>;
            using OUTPUT_SHAPE = typename HEAD_TYPE::OUTPUT_SHAPE;
        };
    public:
        using HEAD_TYPE = typename HeadResolver<HAS_HEAD>::HEAD_TYPE;
        using OUTPUT_SHAPE = typename HeadResolver<HAS_HEAD>::OUTPUT_SHAPE;

        using TYPE_POLICY = typename PIPELINE_TYPE_A::TYPE_POLICY;
    };

    template <typename T_SPEC, bool T_DYNAMIC_ALLOCATION>
    struct ModuleBufferSpecification;
    template <typename T_BUFFER_SPEC>
    struct ModuleBuffer;
    template <typename T_SPEC, bool T_DYNAMIC_ALLOCATION>
    struct ModuleStateSpecification;
    template <typename T_STATE_SPEC>
    struct ModuleState;

    template <typename T_SPEC>
    struct ModuleForward{
        using SPEC = T_SPEC;
        using TYPE_POLICY = typename SPEC::TYPE_POLICY;
        using TI = typename SPEC::TI;
        using INPUT_SHAPE_A = typename SPEC::INPUT_SHAPE_A;
        using INPUT_SHAPE_B = typename SPEC::INPUT_SHAPE_B;
        using INPUT_SHAPE = INPUT_SHAPE_A; // primary input shape (for PPO compatibility)
        using OUTPUT_SHAPE = typename SPEC::OUTPUT_SHAPE;

        typename SPEC::PIPELINE_TYPE_A pipeline_a;
        typename SPEC::PIPELINE_TYPE_B pipeline_b;
        typename SPEC::HEAD_TYPE head;

        template <bool DYNAMIC_ALLOCATION=true>
        using Buffer = ModuleBuffer<ModuleBufferSpecification<SPEC, DYNAMIC_ALLOCATION>>;
        template <bool DYNAMIC_ALLOCATION=true>
        using State = ModuleState<ModuleStateSpecification<SPEC, DYNAMIC_ALLOCATION>>;
    };

    template <typename T_SPEC>
    struct ModuleBackward: public ModuleForward<T_SPEC>{
        using PARENT = ModuleForward<T_SPEC>;
    };

    template <typename T_SPEC>
    struct ModuleGradient: public ModuleBackward<T_SPEC>{
        using PARENT = ModuleBackward<T_SPEC>;
        using TI = typename T_SPEC::TI;
        using T = typename T_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Activation>;
        using OUTPUT_CONTAINER_SHAPE = typename T_SPEC::OUTPUT_SHAPE;
        using OUTPUT_CONTAINER_SPEC = tensor::Specification<T, TI, OUTPUT_CONTAINER_SHAPE, T_SPEC::CAPABILITY::DYNAMIC_ALLOCATION, tensor::RowMajorStride<OUTPUT_CONTAINER_SHAPE>, T_SPEC::CAPABILITY::CONST>;
        using OUTPUT_CONTAINER_TYPE = Tensor<OUTPUT_CONTAINER_SPEC>;
        OUTPUT_CONTAINER_TYPE output;
    };

    template <typename T_SPEC, bool T_DYNAMIC_ALLOCATION>
    struct ModuleBufferSpecification{
        using SPEC = T_SPEC;
        static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
    };

    template <typename T_BUFFER_SPEC>
    struct ModuleBuffer{
        using BUFFER_SPEC = T_BUFFER_SPEC;
        using SPEC = typename BUFFER_SPEC::SPEC;
        using TI = typename SPEC::TI;
        using TYPE_POLICY = typename SPEC::TYPE_POLICY;
        using T = typename TYPE_POLICY::template GET<numeric_types::categories::Activation>;
        static constexpr bool DYNAMIC_ALLOCATION = BUFFER_SPEC::DYNAMIC_ALLOCATION;

        typename SPEC::PIPELINE_TYPE_A::template Buffer<DYNAMIC_ALLOCATION> buffer_a;
        typename SPEC::PIPELINE_TYPE_B::template Buffer<DYNAMIC_ALLOCATION> buffer_b;

        using INTERMEDIATE_A_SPEC = tensor::Specification<T, TI, typename SPEC::OUTPUT_SHAPE_A, DYNAMIC_ALLOCATION, tensor::RowMajorStride<typename SPEC::OUTPUT_SHAPE_A>>;
        using INTERMEDIATE_B_SPEC = tensor::Specification<T, TI, typename SPEC::OUTPUT_SHAPE_B, DYNAMIC_ALLOCATION, tensor::RowMajorStride<typename SPEC::OUTPUT_SHAPE_B>>;
        Tensor<INTERMEDIATE_A_SPEC> intermediate_a;
        Tensor<INTERMEDIATE_B_SPEC> intermediate_b;

        // Concatenated output buffer (used as input to head when HAS_HEAD)
        using CONCAT_OUTPUT_SPEC = tensor::Specification<T, TI, typename SPEC::CONCAT_OUTPUT_SHAPE, DYNAMIC_ALLOCATION, tensor::RowMajorStride<typename SPEC::CONCAT_OUTPUT_SHAPE>>;
        Tensor<CONCAT_OUTPUT_SPEC> concatenated;

        using D_OUTPUT_A_SPEC = tensor::Specification<T, TI, typename SPEC::OUTPUT_SHAPE_A, DYNAMIC_ALLOCATION, tensor::RowMajorStride<typename SPEC::OUTPUT_SHAPE_A>>;
        using D_OUTPUT_B_SPEC = tensor::Specification<T, TI, typename SPEC::OUTPUT_SHAPE_B, DYNAMIC_ALLOCATION, tensor::RowMajorStride<typename SPEC::OUTPUT_SHAPE_B>>;
        Tensor<D_OUTPUT_A_SPEC> d_output_a;
        Tensor<D_OUTPUT_B_SPEC> d_output_b;

        // Head buffer (conditionally present)
    private:
        template <bool ENABLED, typename = void>
        struct HeadBufferResolver{ using type = detail::Empty; };
        template <typename DUMMY>
        struct HeadBufferResolver<true, DUMMY>{ using type = typename SPEC::HEAD_TYPE::template Buffer<DYNAMIC_ALLOCATION>; };
    public:
        typename HeadBufferResolver<SPEC::HAS_HEAD>::type head_buffer;

        // d_concatenated: gradient of loss w.r.t. concatenated output (needed when head present)
        using D_CONCAT_SPEC = tensor::Specification<T, TI, typename SPEC::CONCAT_OUTPUT_SHAPE, DYNAMIC_ALLOCATION, tensor::RowMajorStride<typename SPEC::CONCAT_OUTPUT_SHAPE>>;
        Tensor<D_CONCAT_SPEC> d_concatenated;
    };

    template <typename T_SPEC, bool T_DYNAMIC_ALLOCATION>
    struct ModuleStateSpecification{
        using SPEC = T_SPEC;
        static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
    };

    template <typename T_STATE_SPEC>
    struct ModuleState{
        using STATE_SPEC = T_STATE_SPEC;
        using SPEC = typename STATE_SPEC::SPEC;
        static constexpr bool DYNAMIC_ALLOCATION = STATE_SPEC::DYNAMIC_ALLOCATION;
        typename SPEC::PIPELINE_TYPE_A::template State<DYNAMIC_ALLOCATION> state_a;
        typename SPEC::PIPELINE_TYPE_B::template State<DYNAMIC_ALLOCATION> state_b;
    private:
        template <bool ENABLED, typename = void>
        struct HeadStateResolver{ using type = detail::Empty; };
        template <typename DUMMY>
        struct HeadStateResolver<true, DUMMY>{ using type = typename SPEC::HEAD_TYPE::template State<DYNAMIC_ALLOCATION>; };
    public:
        typename HeadStateResolver<SPEC::HAS_HEAD>::type head_state;
    };

    template <typename CAPABILITY, typename SPEC>
    struct BuildModuleType{
        using FORWARD = ModuleForward<SPEC>;
        using BACKWARD = ModuleBackward<SPEC>;
        using GRADIENT = ModuleGradient<SPEC>;
        using type = utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Forward, FORWARD,
            utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Backward, BACKWARD,
            utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Gradient, GRADIENT, void>>>;
    };

    template <typename CAPABILITY, typename MODULE_A, typename MODULE_B, typename INPUT_SHAPE_A, typename INPUT_SHAPE_B, typename HEAD = void>
    struct Build: BuildModuleType<CAPABILITY, Specification<CAPABILITY, MODULE_A, MODULE_B, INPUT_SHAPE_A, INPUT_SHAPE_B, HEAD>>::type{
        using PARALLEL_SPEC = Specification<CAPABILITY, MODULE_A, MODULE_B, INPUT_SHAPE_A, INPUT_SHAPE_B, HEAD>;
        template <typename NEW_CAPABILITY>
        using CHANGE_CAPABILITY = Build<NEW_CAPABILITY, MODULE_A, MODULE_B, INPUT_SHAPE_A, INPUT_SHAPE_B, HEAD>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif

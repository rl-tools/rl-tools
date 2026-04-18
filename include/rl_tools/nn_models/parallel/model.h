#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_PARALLEL_MODEL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_PARALLEL_MODEL_H

#include "../sequential/model.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn_models::parallel{

    template <typename T_MODULE, typename T_INPUT_SHAPE>
    struct Branch{
        using MODULE = T_MODULE;
        using INPUT_SHAPE = T_INPUT_SHAPE;
    };

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

        struct Empty{};

        template <typename FIRST, typename...>
        struct first_type{ using type = FIRST; };

        template <typename T>
        struct is_input_tuple : utils::typing::false_type {};
        template <typename TI, typename... Types>
        struct is_input_tuple<utils::Tuple<TI, Types...>> : utils::typing::true_type {};
        template <typename TI, typename... Types, template <typename> typename F>
        struct is_input_tuple<utils::MapTuple<utils::Tuple<TI, Types...>, F>> : utils::typing::true_type {};

        template <typename CAPABILITY, typename BRANCH>
        using pipeline_type = typename BRANCH::MODULE::template Layer<CAPABILITY, typename BRANCH::INPUT_SHAPE>;

        template <typename CAPABILITY, typename BRANCH>
        using output_shape = typename pipeline_type<CAPABILITY, BRANCH>::OUTPUT_SHAPE;

        template <typename CAPABILITY, typename... BRANCHES>
        struct ConcatLastDim{
            static constexpr auto VALUE = 0;
        };
        template <typename CAPABILITY, typename BRANCH, typename... REST>
        struct ConcatLastDim<CAPABILITY, BRANCH, REST...>{
            static constexpr auto VALUE = get_last(output_shape<CAPABILITY, BRANCH>{}) + ConcatLastDim<CAPABILITY, REST...>::VALUE;
        };

        template <typename CAPABILITY, typename... BRANCHES>
        struct VerifyShapes{
            static constexpr bool VALID = true;
        };
        template <typename CAPABILITY, typename FIRST, typename SECOND, typename... REST>
        struct VerifyShapes<CAPABILITY, FIRST, SECOND, REST...>{
            using SHAPE_A = output_shape<CAPABILITY, FIRST>;
            using SHAPE_B = output_shape<CAPABILITY, SECOND>;
            static_assert(length(SHAPE_A{}) == length(SHAPE_B{}), "Pipeline output shapes must have same rank");
            static_assert(leading_dims_match<SHAPE_A, SHAPE_B>(), "Pipeline output shapes must have matching leading dimensions");
            static constexpr bool VALID = VerifyShapes<CAPABILITY, FIRST, REST...>::VALID;
        };

        template <typename CAPABILITY>
        struct PipelineMapFactory{
            template <typename BRANCH>
            struct Map{
                using CONTENT = pipeline_type<CAPABILITY, BRANCH>;
            };
        };

        template <typename BRANCH>
        struct InputShapeMap{
            using CONTENT = typename BRANCH::INPUT_SHAPE;
        };

        template <typename CAPABILITY, bool DYNAMIC_ALLOCATION, typename TYPE_POLICY>
        struct TensorMapFactory{
            template <typename BRANCH>
            struct Map{
                using PIPELINE = pipeline_type<CAPABILITY, BRANCH>;
                using OUTPUT_SHAPE = typename PIPELINE::OUTPUT_SHAPE;
                using T = typename TYPE_POLICY::template GET<numeric_types::categories::Activation>;
                using TI = typename OUTPUT_SHAPE::TI;
                using CONTENT = Tensor<tensor::Specification<T, TI, OUTPUT_SHAPE, DYNAMIC_ALLOCATION, tensor::RowMajorStride<OUTPUT_SHAPE>>>;
            };
        };

        template <typename CAPABILITY, bool DYNAMIC_ALLOCATION>
        struct SubBufferMapFactory{
            template <typename BRANCH>
            struct Map{
                using CONTENT = typename pipeline_type<CAPABILITY, BRANCH>::template Buffer<DYNAMIC_ALLOCATION>;
            };
        };

        template <typename CAPABILITY, bool DYNAMIC_ALLOCATION>
        struct SubStateMapFactory{
            template <typename BRANCH>
            struct Map{
                using CONTENT = typename pipeline_type<CAPABILITY, BRANCH>::template State<DYNAMIC_ALLOCATION>;
            };
        };
    }

    template <typename T_SPEC, bool T_DYNAMIC_ALLOCATION>
    struct ModuleBufferSpecification;
    template <typename T_BUFFER_SPEC>
    struct ModuleBuffer;
    template <typename T_SPEC, bool T_DYNAMIC_ALLOCATION>
    struct ModuleStateSpecification;
    template <typename T_STATE_SPEC>
    struct ModuleState;

    template <typename T_CAPABILITY, typename T_HEAD, typename... T_BRANCHES>
    struct Specification{
        using CAPABILITY = T_CAPABILITY;
        using HEAD_MODULE = T_HEAD;
        static constexpr bool HAS_HEAD = !utils::typing::is_same_v<HEAD_MODULE, void>;

        using FIRST_BRANCH = typename detail::first_type<T_BRANCHES...>::type;
        using TI = typename FIRST_BRANCH::INPUT_SHAPE::TI;
        static constexpr TI NUM_BRANCHES = sizeof...(T_BRANCHES);
        static_assert(NUM_BRANCHES >= 1, "parallel model requires at least one branch");

        using BRANCH_TUPLE = utils::Tuple<TI, T_BRANCHES...>;
        using PIPELINES_TYPE = utils::MapTuple<BRANCH_TUPLE, detail::PipelineMapFactory<CAPABILITY>::template Map>;
        using INPUT_SHAPES = utils::MapTuple<BRANCH_TUPLE, detail::InputShapeMap>;

        static constexpr bool _SHAPES_VALID = detail::VerifyShapes<CAPABILITY, T_BRANCHES...>::VALID;

        using FIRST_PIPELINE = detail::pipeline_type<CAPABILITY, FIRST_BRANCH>;
        using FIRST_OUTPUT_SHAPE = typename FIRST_PIPELINE::OUTPUT_SHAPE;
        static constexpr auto RANK = length(FIRST_OUTPUT_SHAPE{});
        static constexpr TI CONCAT_LAST_DIM = detail::ConcatLastDim<CAPABILITY, T_BRANCHES...>::VALUE;
        using CONCAT_OUTPUT_SHAPE = tensor::Replace<FIRST_OUTPUT_SHAPE, CONCAT_LAST_DIM, RANK - 1>;

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
        using TYPE_POLICY = typename FIRST_PIPELINE::TYPE_POLICY;
    };

    template <typename T_SPEC>
    struct ModuleForward{
        using SPEC = T_SPEC;
        using TYPE_POLICY = typename SPEC::TYPE_POLICY;
        using TI = typename SPEC::TI;
        using INPUT_SHAPES = typename SPEC::INPUT_SHAPES;
        using OUTPUT_SHAPE = typename SPEC::OUTPUT_SHAPE;

        typename SPEC::PIPELINES_TYPE pipelines;
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

        utils::MapTuple<typename SPEC::BRANCH_TUPLE, detail::SubBufferMapFactory<typename SPEC::CAPABILITY, DYNAMIC_ALLOCATION>::template Map> sub_buffers;
        utils::MapTuple<typename SPEC::BRANCH_TUPLE, detail::TensorMapFactory<typename SPEC::CAPABILITY, DYNAMIC_ALLOCATION, TYPE_POLICY>::template Map> intermediates;
        utils::MapTuple<typename SPEC::BRANCH_TUPLE, detail::TensorMapFactory<typename SPEC::CAPABILITY, DYNAMIC_ALLOCATION, TYPE_POLICY>::template Map> d_outputs;

        using CONCAT_SPEC = tensor::Specification<T, TI, typename SPEC::CONCAT_OUTPUT_SHAPE, DYNAMIC_ALLOCATION, tensor::RowMajorStride<typename SPEC::CONCAT_OUTPUT_SHAPE>>;
        Tensor<CONCAT_SPEC> concatenated;
        Tensor<CONCAT_SPEC> d_concatenated;

    private:
        template <bool ENABLED, typename = void>
        struct HeadBufferResolver{ using type = detail::Empty; };
        template <typename DUMMY>
        struct HeadBufferResolver<true, DUMMY>{ using type = typename SPEC::HEAD_TYPE::template Buffer<DYNAMIC_ALLOCATION>; };
    public:
        typename HeadBufferResolver<SPEC::HAS_HEAD>::type head_buffer;
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

        utils::MapTuple<typename SPEC::BRANCH_TUPLE, detail::SubStateMapFactory<typename SPEC::CAPABILITY, DYNAMIC_ALLOCATION>::template Map> states;

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

    template <typename CAPABILITY, typename HEAD, typename... BRANCHES>
    struct Build: BuildModuleType<CAPABILITY, Specification<CAPABILITY, HEAD, BRANCHES...>>::type{
        using PARALLEL_SPEC = Specification<CAPABILITY, HEAD, BRANCHES...>;
        template <typename NEW_CAPABILITY>
        using CHANGE_CAPABILITY = Build<NEW_CAPABILITY, HEAD, BRANCHES...>;
        template <typename TI, TI BATCH_SIZE>
        struct CHANGE_BATCH_SIZE_IMPL{
            template <typename BRANCH>
            using UpdatedBranch = Branch<typename BRANCH::MODULE, tensor::Replace<typename BRANCH::INPUT_SHAPE, BATCH_SIZE, 1>>;
            using CHANGE_BATCH_SIZE = Build<CAPABILITY, HEAD, UpdatedBranch<BRANCHES>...>;
        };
        template <typename TI, TI BATCH_SIZE>
        using CHANGE_BATCH_SIZE = typename CHANGE_BATCH_SIZE_IMPL<TI, BATCH_SIZE>::CHANGE_BATCH_SIZE;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif

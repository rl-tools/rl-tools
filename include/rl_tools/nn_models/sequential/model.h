#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_SEQUENTIAL_MODEL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_SEQUENTIAL_MODEL_H

#include "../../utils/generic/typing.h"
#include "../../utils/generic/tuple/tuple.h"
#include "../../nn/nn.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn_models::sequential{
    struct OutputModule{
        struct CONTENT{
            using INPUT_SHAPE = tensor::Shape<decltype(0), 0>;
        };
        template <typename>
        using CHANGE_CAPABILITY = OutputModule;
    };

    template <typename T_CONTENT, typename T_INPUT_SHAPE, typename T_OUTPUT_SHAPE>
    struct LayerSpecification {
        using CONTENT = T_CONTENT;
        using INPUT_SHAPE = T_INPUT_SHAPE;
        using OUTPUT_SHAPE = T_OUTPUT_SHAPE;
        using TI = typename CONTENT::TI;
        using TYPE_POLICY = typename CONTENT::TYPE_POLICY;
    };

    template <typename TUPLE>
    struct tuple_size;

    template <typename TI, typename... TYPES>
    struct tuple_size<utils::Tuple<TI, TYPES...>> {
        static constexpr TI value = sizeof...(TYPES);
    };

    template <typename TUPLE, typename T>
    struct tuple_append;

    template <typename TI, typename... TYPES, typename T>
    struct tuple_append<utils::Tuple<TI, TYPES...>, T> {
        using type = utils::Tuple<TI, TYPES..., T>;
    };

    template <typename TUPLE, typename T>
    using tuple_append_t = typename tuple_append<TUPLE, T>::type;

    template <auto INDEX, typename TUPLE>
    struct tuple_element;

    namespace detail {
        template <typename TI, TI... Is>
        struct index_sequence {};

        template <typename TI, bool DONE, TI N, TI... Is>
        struct make_index_sequence_impl;
        template <typename TI, TI N, TI... Is>
        struct make_index_sequence_impl<TI, true, N, Is...> {
            using type = index_sequence<TI, Is...>;
        };
        template <typename TI, TI N, TI... Is>
        struct make_index_sequence_impl<TI, false, N, Is...> {
            using type = typename make_index_sequence_impl<TI, N - 1 == 0, N - 1, N - 1, Is...>::type;
        };
        template <typename TI, TI N>
        using make_index_sequence = typename make_index_sequence_impl<TI, N == 0, N>::type;

        template <typename TI, TI Index, typename T>
        struct TupleLeaf {
            using type = T;
        };

        template <typename SEQ, typename TI, typename... Ts>
        struct TupleIndex;
        template <typename TI, TI... Is, typename... Ts>
        struct TupleIndex<index_sequence<TI, Is...>, TI, Ts...> : TupleLeaf<TI, Is, Ts>... {};

        template <typename TI, TI I, typename T>
        TupleLeaf<TI, I, T> select_leaf(const TupleLeaf<TI, I, T>&);
    }

    template <auto INDEX, typename TI, typename... TYPES>
    struct tuple_element<INDEX, utils::Tuple<TI, TYPES...>> {
        static_assert(static_cast<TI>(INDEX) < sizeof...(TYPES), "tuple_element index out of bounds");
        using Indexed = detail::TupleIndex<detail::make_index_sequence<TI, sizeof...(TYPES)>, TI, TYPES...>;
        using type = typename decltype(detail::select_leaf<TI, static_cast<TI>(INDEX)>(Indexed{}))::type;
    };

    template <typename... T_CONTENTS>
    struct Module {};

    namespace detail {
        template <typename...>
        struct has_real_layers : utils::typing::false_type {};
        template <typename T, typename... Ts>
        struct has_real_layers<T, Ts...> : utils::typing::true_type {};
        template <typename... Ts>
        struct has_real_layers<OutputModule, Ts...> : utils::typing::false_type {};
        template <typename... NESTED, typename... REST>
        struct has_real_layers<Module<NESTED...>, REST...> : has_real_layers<NESTED..., REST...> {};
    }

    template <typename CAPABILITY, typename T_MODULE, typename INPUT_SHAPE, typename ACCUMULATOR, typename TI, TI CURRENT_MAX>
    struct BuildLayerSpecsImpl;

    template <typename CAPABILITY, typename INPUT_SHAPE, typename ACCUMULATOR, typename TI, TI CURRENT_MAX>
    struct BuildLayerSpecsImpl<CAPABILITY, Module<>, INPUT_SHAPE, ACCUMULATOR, TI, CURRENT_MAX> {
        using LAYER_SPECS = ACCUMULATOR;
        using FINAL_OUTPUT_SHAPE = INPUT_SHAPE;
        static constexpr TI MAX_HIDDEN_DIM = CURRENT_MAX;
    };

    template <typename CAPABILITY, typename... TAIL, typename INPUT_SHAPE, typename ACCUMULATOR, typename TI, TI CURRENT_MAX>
    struct BuildLayerSpecsImpl<CAPABILITY, Module<OutputModule, TAIL...>, INPUT_SHAPE, ACCUMULATOR, TI, CURRENT_MAX> {
        static_assert(sizeof...(TAIL) == 0, "OutputModule must be the last element in a Module chain");
        using LAYER_SPECS = ACCUMULATOR;
        using FINAL_OUTPUT_SHAPE = INPUT_SHAPE;
        static constexpr TI MAX_HIDDEN_DIM = CURRENT_MAX;
    };

    template <typename CAPABILITY, typename HEAD, typename... TAIL, typename INPUT_SHAPE, typename ACCUMULATOR, typename TI, TI CURRENT_MAX>
    struct BuildLayerSpecsImpl<CAPABILITY, Module<HEAD, TAIL...>, INPUT_SHAPE, ACCUMULATOR, TI, CURRENT_MAX> {
        using CONTENT = typename HEAD::template Layer<CAPABILITY, INPUT_SHAPE>;
        using OUTPUT_SHAPE = typename CONTENT::SPEC::OUTPUT_SHAPE;
        using LAYER_SPEC = LayerSpecification<CONTENT, INPUT_SHAPE, OUTPUT_SHAPE>;
        static constexpr TI NEW_MAX = detail::has_real_layers<TAIL...>::value ?
            (CURRENT_MAX > product(OUTPUT_SHAPE{}) ? CURRENT_MAX : product(OUTPUT_SHAPE{})) : CURRENT_MAX;
        using NEXT_ACCUMULATOR = tuple_append_t<ACCUMULATOR, LAYER_SPEC>;
        using NEXT = BuildLayerSpecsImpl<CAPABILITY, Module<TAIL...>, OUTPUT_SHAPE, NEXT_ACCUMULATOR, TI, NEW_MAX>;
        using LAYER_SPECS = typename NEXT::LAYER_SPECS;
        using FINAL_OUTPUT_SHAPE = typename NEXT::FINAL_OUTPUT_SHAPE;
        static constexpr TI MAX_HIDDEN_DIM = NEXT::MAX_HIDDEN_DIM;
    };

    template <typename CAPABILITY, typename... NESTED, typename... TAIL, typename INPUT_SHAPE, typename ACCUMULATOR, typename TI, TI CURRENT_MAX>
    struct BuildLayerSpecsImpl<CAPABILITY, Module<Module<NESTED...>, TAIL...>, INPUT_SHAPE, ACCUMULATOR, TI, CURRENT_MAX>
        : BuildLayerSpecsImpl<CAPABILITY, Module<NESTED..., TAIL...>, INPUT_SHAPE, ACCUMULATOR, TI, CURRENT_MAX> {};

    template <typename TI, typename SPEC, auto INDEX = 0>
    constexpr TI find_max_hiddend_dim(TI current_max = 0){
        if constexpr(INDEX + 1 >= SPEC::NUM_LAYERS){
            return current_max;
        }
        else{
            using LAYER_SPEC = typename tuple_element<INDEX, typename SPEC::LAYER_SPECS>::type;
            constexpr TI OUT_DIM = product(typename LAYER_SPEC::OUTPUT_SHAPE{});
            TI next_max = current_max > OUT_DIM ? current_max : OUT_DIM;
            return find_max_hiddend_dim<TI, SPEC, INDEX + 1>(next_max);
        }
    }

    template <typename T_CAPABILITY, typename T_MODULE, typename T_INPUT_SHAPE, typename T_LAYER_SPECS, typename T_OUTPUT_SHAPE, auto T_MAX_HIDDEN_DIM>
    struct Specification{
        using CAPABILITY = T_CAPABILITY;
        using MODULE_CHAIN = T_MODULE;
        using INPUT_SHAPE = T_INPUT_SHAPE;
        using OUTPUT_SHAPE = T_OUTPUT_SHAPE;
        using LAYER_SPECS = T_LAYER_SPECS;
        using TI = typename INPUT_SHAPE::TI;
        static constexpr TI NUM_LAYERS = tuple_size<LAYER_SPECS>::value;
        static constexpr TI MAX_HIDDEN_DIM = static_cast<TI>(T_MAX_HIDDEN_DIM);
        using FIRST_LAYER_SPEC = typename tuple_element<0, LAYER_SPECS>::type;
        using TYPE_POLICY = typename FIRST_LAYER_SPEC::TYPE_POLICY;
        using ORIGINAL_ROOT = T_MODULE;
        using CONTENT = typename FIRST_LAYER_SPEC::CONTENT;
        using NEXT_MODULE = OutputModule;
    };

    template <typename CAPABILITY, typename MODULE, typename INPUT_SHAPE>
    struct BuildSpecification {
        using TI = typename INPUT_SHAPE::TI;
        using BUILDER = BuildLayerSpecsImpl<CAPABILITY, MODULE, INPUT_SHAPE, utils::Tuple<TI>, TI, static_cast<TI>(0)>;
        using type = Specification<CAPABILITY, MODULE, INPUT_SHAPE, typename BUILDER::LAYER_SPECS, typename BUILDER::FINAL_OUTPUT_SHAPE, BUILDER::MAX_HIDDEN_DIM>;
    };

    template <typename T_SPEC>
    struct ModuleForward;

    template <typename T_SPEC, bool T_DYNAMIC_ALLOCATION>
    struct ContentStateSpecification {
        using SPEC = T_SPEC;
        static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
    };

    template <bool DYNAMIC_ALLOCATION>
    struct ContentStateMapFactory {
        template <typename LAYER_SPEC>
        struct Map {
            using CONTENT = typename LAYER_SPEC::CONTENT::template State<DYNAMIC_ALLOCATION>;
        };
    };

    template <typename T_SPEC>
    struct ContentState{
        using SPEC = T_SPEC;
        using STATE_LIST = utils::MapTuple<typename SPEC::SPEC::LAYER_SPECS, ContentStateMapFactory<SPEC::DYNAMIC_ALLOCATION>::template Map>;
        STATE_LIST states;
    };

    template <typename T_SPEC, bool T_DYNAMIC_ALLOCATION = true>
    struct ModuleStateSpecification {
        using SPEC = T_SPEC;
        static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
        using CONTENT_BUFFER_SPEC = ContentStateSpecification<SPEC, DYNAMIC_ALLOCATION>;
    };

    template <typename T_BUFFER_SPEC>
    struct ModuleState{
        using BUFFER_SPEC = T_BUFFER_SPEC;
        using CONTENT_STATE = ContentState<typename BUFFER_SPEC::CONTENT_BUFFER_SPEC>;
        CONTENT_STATE content_state;
    };

    template <typename T_SPEC, bool T_DYNAMIC_ALLOCATION>
    struct ContentBufferSpecification {
        using SPEC = T_SPEC;
        static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
    };

    template <bool DYNAMIC_ALLOCATION>
    struct ContentBufferMapFactory {
        template <typename LAYER_SPEC>
        struct Map {
            using CONTENT = typename LAYER_SPEC::CONTENT::template Buffer<DYNAMIC_ALLOCATION>;
        };
    };

    template <typename T_SPEC>
    struct ContentBuffer{
        using SPEC = T_SPEC;
        using BUFFER_LIST = utils::MapTuple<typename SPEC::SPEC::LAYER_SPECS, ContentBufferMapFactory<SPEC::DYNAMIC_ALLOCATION>::template Map>;
        BUFFER_LIST buffers;
    };

    template <typename T_SPEC, bool T_DYNAMIC_ALLOCATION = true>
    struct ModuleBufferSpecification {
        using SPEC = T_SPEC;
        using TYPE_POLICY = typename SPEC::TYPE_POLICY;
        using TI = typename SPEC::TI;
        static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
        using CONTENT_BUFFER_SPEC = ContentBufferSpecification<SPEC, DYNAMIC_ALLOCATION>;
    };

    template <typename T_BUFFER_SPEC>
    struct ModuleBuffer{
        using BUFFER_SPEC = T_BUFFER_SPEC;
        using SPEC = typename BUFFER_SPEC::SPEC;
        using TYPE_POLICY = typename SPEC::TYPE_POLICY;
        using T_ACCUMULATOR = typename TYPE_POLICY::template GET<numeric_types::categories::Accumulator>;
        using TI = typename SPEC::TI;
        using TICK_TOCK_CONTAINER_SHAPE = tensor::Shape<TI, SPEC::MAX_HIDDEN_DIM>;
        using TICK_TOCK_CONTAINER_SPEC = tensor::Specification<T_ACCUMULATOR, TI, TICK_TOCK_CONTAINER_SHAPE, BUFFER_SPEC::DYNAMIC_ALLOCATION, tensor::RowMajorStride<TICK_TOCK_CONTAINER_SHAPE>>;
        using TICK_TOCK_CONTAINER_TYPE = Tensor<TICK_TOCK_CONTAINER_SPEC>;
        TICK_TOCK_CONTAINER_TYPE tick;
        TICK_TOCK_CONTAINER_TYPE tock;
        using CONTENT_BUFFER = ContentBuffer<typename BUFFER_SPEC::CONTENT_BUFFER_SPEC>;
        CONTENT_BUFFER content_buffer;
    };

    template <typename LAYER_SPEC>
    struct LayerContentMap {
        using CONTENT = typename LAYER_SPEC::CONTENT;
    };

    template <typename T_SPEC>
    struct ModuleForward{
        using SPEC = T_SPEC;
        using TYPE_POLICY = typename SPEC::TYPE_POLICY;
        using TI = typename SPEC::TI;
        using ORIGINAL_ROOT = typename SPEC::ORIGINAL_ROOT;
        using LAYERS = utils::MapTuple<typename SPEC::LAYER_SPECS, LayerContentMap>;
        LAYERS content;

        using INPUT_SHAPE = typename SPEC::INPUT_SHAPE;
        using OUTPUT_SHAPE = typename SPEC::OUTPUT_SHAPE;

        template <bool DYNAMIC_ALLOCATION=true>
        using State = ModuleState<ModuleStateSpecification<SPEC, DYNAMIC_ALLOCATION>>;
        template <bool DYNAMIC_ALLOCATION=true>
        using Buffer = ModuleBuffer<ModuleBufferSpecification<SPEC, DYNAMIC_ALLOCATION>>;
    };

    template <typename T_SPEC>
    struct ModuleBackward: public ModuleForward<T_SPEC>{
        using PARENT = ModuleForward<T_SPEC>;
    };

    template <typename T_SPEC>
    struct ModuleGradient: public ModuleBackward<T_SPEC>{
        using PARENT = ModuleBackward<T_SPEC>;
        using TI = typename T_SPEC::TI;
    };

    template <typename CAPABILITY, typename ROOT_SPEC>
    struct BuildModuleType;

    template <typename CAPABILITY, typename ROOT_SPEC>
    struct BuildModuleType {
        using FORWARD = ModuleForward<ROOT_SPEC>;
        using BACKWARD = ModuleBackward<ROOT_SPEC>;
        using GRADIENT = ModuleGradient<ROOT_SPEC>;
        using type = utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Forward, FORWARD,
            utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Backward, BACKWARD,
            utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Gradient, GRADIENT, void>>>;
    };

    template <typename CAPABILITY, typename MODULE, typename INPUT_SHAPE>
    struct Build: BuildModuleType<CAPABILITY, typename BuildSpecification<CAPABILITY, MODULE, INPUT_SHAPE>::type>::type{
        template <typename TI, TI BATCH_SIZE>
        struct CHANGE_BATCH_SIZE_IMPL{
            using NEW_INPUT_SHAPE = tensor::Replace<INPUT_SHAPE, BATCH_SIZE, 1>;
            using CHANGE_BATCH_SIZE = Build<CAPABILITY, MODULE, NEW_INPUT_SHAPE>;
        };
        template <typename TI, TI BATCH_SIZE>
        using CHANGE_BATCH_SIZE = typename CHANGE_BATCH_SIZE_IMPL<TI, BATCH_SIZE>::CHANGE_BATCH_SIZE;

        template <typename TI, TI SEQUENCE_LENGTH>
        struct CHANGE_SEQUENCE_LENGTH_IMPL{
            using NEW_INPUT_SHAPE = tensor::Replace<INPUT_SHAPE, SEQUENCE_LENGTH, 0>;
            using CHANGE_SEQUENCE_LENGTH = Build<CAPABILITY, MODULE, NEW_INPUT_SHAPE>;
        };
        template <typename TI, TI SEQUENCE_LENGTH>
        using CHANGE_SEQUENCE_LENGTH = typename CHANGE_SEQUENCE_LENGTH_IMPL<TI, SEQUENCE_LENGTH>::CHANGE_SEQUENCE_LENGTH;

        template <typename NEW_CAPABILITY>
        using CHANGE_CAPABILITY = Build<NEW_CAPABILITY, MODULE, INPUT_SHAPE>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif

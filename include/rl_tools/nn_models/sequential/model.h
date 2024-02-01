#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_SEQUENTIAL_MODEL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_SEQUENTIAL_MODEL_H

#include "../../utils/generic/typing.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn_models::sequential{
    struct OutputModule{
        struct CONTENT{
            static constexpr auto INPUT_DIM = 0;
            static constexpr auto BATCH_SIZE = 0;
        };
    };

    // Required fields on CONTENT:
    // compile-time types
    //     T
    //     TI
    // compile-time constants
    //     INPUT_DIM
    //     OUTPUT_DIM
    //     BATCH_SIZE
    // run-time containers
    //     output (just required for forward and backward)
    // containers
    //     operations
    //     malloc
    //     free
    //     init_weights
    //     forward

    template <typename SPEC>
    constexpr auto find_output_dim() {
        if constexpr (utils::typing::is_same_v<typename SPEC::NEXT_MODULE, OutputModule>){
            return SPEC::CONTENT::OUTPUT_DIM;
        } else {
            return find_output_dim<typename SPEC::NEXT_MODULE>();
        }
    }
    template <typename TI, typename SPEC>
    constexpr auto find_max_hiddend_dim(TI current_max = 0){
        current_max = current_max > SPEC::CONTENT::OUTPUT_DIM ? current_max : SPEC::CONTENT::OUTPUT_DIM;
        if constexpr (utils::typing::is_same_v<typename SPEC::NEXT_MODULE, OutputModule>){
            return 0;
        } else {
            auto max_downstream = find_max_hiddend_dim<TI, typename SPEC::NEXT_MODULE>();
            return max_downstream > current_max ? max_downstream : current_max;
        }
    }
    template <typename MODULE>
    constexpr bool check_batch_size_consistency_f(){
        if constexpr (utils::typing::is_same_v<typename MODULE::NEXT_MODULE, OutputModule>){
            return true;
        } else {
            return MODULE::CONTENT::BATCH_SIZE == MODULE::NEXT_MODULE::CONTENT::BATCH_SIZE && check_batch_size_consistency_f<typename MODULE::NEXT_MODULE>();
        }
    }

    template <typename MODULE>
    constexpr bool check_batch_size_consistency = check_batch_size_consistency_f<MODULE>();

    template <typename SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    constexpr bool check_input_output_f(){
        static_assert(INPUT_SPEC::COLS == SPEC::CONTENT::INPUT_DIM);
        static_assert(OUTPUT_SPEC::ROWS == INPUT_SPEC::ROWS);
        static_assert(OUTPUT_SPEC::COLS == find_output_dim<SPEC>());
        static_assert(OUTPUT_SPEC::ROWS == INPUT_SPEC::ROWS);
        return true;
    }
    template <typename SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    constexpr bool check_input_output = check_input_output_f<SPEC, INPUT_SPEC, OUTPUT_SPEC>();


    template <typename BUFFER_SPEC, typename MODULE_SPEC>
    constexpr bool buffer_compatible = BUFFER_SPEC::SPEC::MAX_HIDDEN_DIM >= MODULE_SPEC::MAX_HIDDEN_DIM;

    template <typename T_CONTENT, typename T_NEXT_MODULE = OutputModule>
    struct Specification{
        using CONTENT = T_CONTENT;
        using NEXT_MODULE = T_NEXT_MODULE;
        using T = typename CONTENT::T;
        using TI = typename CONTENT::TI;
        using CONTAINER_TYPE_TAG = typename CONTENT::CONTAINER_TYPE_TAG;
        static constexpr TI INPUT_DIM = CONTENT::INPUT_DIM;
        static constexpr TI OUTPUT_DIM = find_output_dim<Specification<T_CONTENT, T_NEXT_MODULE>>();
        static constexpr TI MAX_HIDDEN_DIM = find_max_hiddend_dim<typename CONTENT::TI, Specification<T_CONTENT, T_NEXT_MODULE>>();
        static_assert(utils::typing::is_same_v<NEXT_MODULE, OutputModule> || CONTENT::OUTPUT_DIM == NEXT_MODULE::CONTENT::INPUT_DIM);
    };

    template <typename T_SPEC, typename T_SPEC::TI T_BATCH_SIZE, typename T_CONTAINER_TYPE_TAG, typename T_MEMORY_LAYOUT>
    struct ModuleBufferSpecification {
        using SPEC = T_SPEC;
        using TI = typename SPEC::TI;
        static constexpr TI BATCH_SIZE = T_BATCH_SIZE;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
        using MEMORY_LAYOUT = T_MEMORY_LAYOUT;
    };
    template <typename T_BUFFER_SPEC>
    struct ModuleBuffer{
        using BUFFER_SPEC = T_BUFFER_SPEC;
        using SPEC = typename BUFFER_SPEC::SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI BATCH_SIZE = T_BUFFER_SPEC::BATCH_SIZE;
        using TICK_TOCK_CONTAINER_SPEC = matrix::Specification<T, TI, BATCH_SIZE, SPEC::MAX_HIDDEN_DIM, typename BUFFER_SPEC::MEMORY_LAYOUT>;
        using TICK_TOCK_CONTAINER_TYPE = typename BUFFER_SPEC::CONTAINER_TYPE_TAG::template type<TICK_TOCK_CONTAINER_SPEC>;
        TICK_TOCK_CONTAINER_TYPE tick;
        TICK_TOCK_CONTAINER_TYPE tock;
    };
    template <typename T_SPEC>
    struct Module{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using CONTENT = typename SPEC::CONTENT;
        using NEXT_MODULE = typename SPEC::NEXT_MODULE;
        CONTENT content;
        NEXT_MODULE next_module;

        static constexpr auto INPUT_DIM = SPEC::INPUT_DIM;
        static constexpr auto OUTPUT_DIM = SPEC::OUTPUT_DIM;

        template <typename SPEC::TI BATCH_SIZE, typename CONTAINER_TYPE_TAG=typename SPEC::CONTAINER_TYPE_TAG, typename MEMORY_LAYOUT = matrix::layouts::DEFAULT<typename SPEC::TI>>
        using Buffer = ModuleBuffer<ModuleBufferSpecification<SPEC, BATCH_SIZE, CONTAINER_TYPE_TAG, MEMORY_LAYOUT>>;
    };

    namespace interface{
        template <typename T_CONTENT, typename T_NEXT_MODULE = OutputModule>
        struct Module: rl_tools::nn_models::sequential::Module<Specification<T_CONTENT, T_NEXT_MODULE>>{};
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif

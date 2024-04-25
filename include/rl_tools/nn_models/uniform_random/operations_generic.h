#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_UNIFORM_RANDOM_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_UNIFORM_RANDOM_OPERATIONS_GENERIC_H
#include "model.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG>
    void evaluate(const DEVICE& device, nn_models::UniformRandom<SPEC>, Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn_models::uniform_random::Buffer, RNG& rng){
        static_assert(SPEC::OUTPUT_DIM == OUTPUT_SPEC::COLS, "Output dimension mismatch");
        static_assert(SPEC::INPUT_DIM == INPUT_SPEC::COLS, "Input dimension mismatch");
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        for(TI row_i = 0; row_i < OUTPUT_SPEC::ROWS; row_i++){
            for(TI col_i = 0; col_i < OUTPUT_SPEC::COLS; col_i++){
                T value = 0;
                if(SPEC::RANGE == nn_models::uniform_random::Range::MINUS_ONE_TO_ONE){
                    value = random::uniform_real_distribution(device.random, (T)-1, (T)1, rng);
                }else if(SPEC::RANGE == nn_models::uniform_random::Range::ZERO_TO_ONE){
                    value = random::uniform_real_distribution(device.random, (T)0, (T)1, rng);
                }
                set(output, row_i, col_i, value);
            }
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif

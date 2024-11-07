#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_UTILS_POLYAK_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_UTILS_POLYAK_OPERATIONS_GENERIC_H


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::utils::polyak {
    // todo: polyak factor as template parameter (reciprocal INT e.g.)
    template<typename DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void update(DEVICE& device, const  Matrix<SOURCE_SPEC>& source, Matrix<TARGET_SPEC>& target, const typename SOURCE_SPEC::T polyak, bool clip = false, typename SOURCE_SPEC::T clip_value = 1){
        static_assert(containers::check_structure<SOURCE_SPEC, TARGET_SPEC>);
        using SPEC = SOURCE_SPEC;
        using T = typename SPEC::T;
        for(typename DEVICE::index_t i = 0; i < SPEC::ROWS; i++) {
            for(typename DEVICE::index_t j = 0; j < SPEC::COLS; j++) {
                T source_value = get(source, i, j);
                if(clip){
                    source_value = math::clamp(device.math, source_value, -clip_value, clip_value);
                }
                set(target, i, j, polyak * get(target, i, j) + (1 - polyak) * source_value);
            }
        }
    }

    template<typename DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void update_squared(DEVICE& device, const  Matrix<SOURCE_SPEC>& source, Matrix<TARGET_SPEC>& target, const typename SOURCE_SPEC::T polyak, bool clip = false, typename SOURCE_SPEC::T clip_value = 1) {
        static_assert(containers::check_structure<SOURCE_SPEC, TARGET_SPEC>);
        using SPEC = SOURCE_SPEC;
        using T = typename SPEC::T;
        for(typename DEVICE::index_t i = 0; i < SPEC::ROWS; i++) {
            for(typename DEVICE::index_t j = 0; j < SPEC::COLS; j++) {
                T source_value = get(source, i, j);
                if(clip){
                    source_value = math::clamp(device.math, source_value, -clip_value, clip_value);
                }
                set(target, i, j, polyak * get(target, i, j) + (1 - polyak) * source_value * source_value);
            }
        }
    }
    namespace binary_kernels{
        template <typename T>
        struct PolyakParameters{
            T polyak;
            bool clip;
            T clip_value;
        };
        template <typename T>
        T update(T source, T target, const PolyakParameters<T>& p){
            if(p.clip) {
                source = source > p.clip_value ? p.clip_value : (source < -p.clip_value ? -p.clip_value : source);
            }
            return p.polyak * target + (1 - p.polyak) * source;
        }
        template <typename T>
        T update_squared(T source, T target, const PolyakParameters<T>& p){
            if(p.clip) {
                source = source > p.clip_value ? p.clip_value : (source < -p.clip_value ? -p.clip_value : source);
            }
            return p.polyak * target + (1 - p.polyak) * source * source;
        }
    }
    template<typename DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void update(DEVICE& device, const  Tensor<SOURCE_SPEC>& source, Tensor<TARGET_SPEC>& target, const typename SOURCE_SPEC::T polyak, const bool clip = false, typename SOURCE_SPEC::T clip_value = 1) {
        using T = typename SOURCE_SPEC::T;
        tensor::Operation<binary_kernels::update<T>, binary_kernels::PolyakParameters<T>> params{};
        params.parameter.polyak = polyak;
        params.parameter.clip = clip;
        params.parameter.clip_value = clip_value;
        binary_operation(device, params, source, target);
    }

    template<typename DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void update_squared(DEVICE& device, const  Tensor<SOURCE_SPEC>& source, Tensor<TARGET_SPEC>& target, const typename SOURCE_SPEC::T polyak, const bool clip = false, typename SOURCE_SPEC::T clip_value = 1) {
        using T = typename SOURCE_SPEC::T;
        tensor::Operation<binary_kernels::update_squared<T>, binary_kernels::PolyakParameters<T>> params{};
        params.parameter.polyak = polyak;
        params.parameter.clip = clip;
        params.parameter.clip_value = clip_value;
        binary_operation(device, params, source, target);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif
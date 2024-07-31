#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_UTILS_POLYAK_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_UTILS_POLYAK_OPERATIONS_GENERIC_H


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::utils::polyak {
    // todo: polyak factor as template parameter (reciprocal INT e.g.)
    template<typename DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void update(DEVICE& dev, const  Matrix<SOURCE_SPEC>& source, Matrix<TARGET_SPEC>& target, const typename SOURCE_SPEC::T polyak) {
        static_assert(containers::check_structure<SOURCE_SPEC, TARGET_SPEC>);
        using SPEC = SOURCE_SPEC;
        for(typename DEVICE::index_t i = 0; i < SPEC::ROWS; i++) {
            for(typename DEVICE::index_t j = 0; j < SPEC::COLS; j++) {
                set(target, i, j, polyak * get(target, i, j) + (1 - polyak) * get(source, i, j));
            }
        }
    }

    template<typename DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void update_squared(DEVICE& dev, const  Matrix<SOURCE_SPEC>& source, Matrix<TARGET_SPEC>& target, const typename SOURCE_SPEC::T polyak) {
        static_assert(containers::check_structure<SOURCE_SPEC, TARGET_SPEC>);
        using SPEC = SOURCE_SPEC;
        for(typename DEVICE::index_t i = 0; i < SPEC::ROWS; i++) {
            for(typename DEVICE::index_t j = 0; j < SPEC::COLS; j++) {
                typename SPEC::T s = get(source, i, j);
                set(target, i, j, polyak * get(target, i, j) + (1 - polyak) * s * s);
            }
        }
    }
    namespace binary_kernels{
        template <typename T>
        struct PolyakParameters{
            T polyak;
        };
        template <typename T>
        T update(T source, T target, const PolyakParameters<T>& p){
            return p.polyak * target + (1 - p.polyak) * source;
        }
        template <typename T>
        T update_squared(T source, T target, const PolyakParameters<T>& p){
            return p.polyak * target + (1 - p.polyak) * source * source;
        }
    }
    template<typename DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void update(DEVICE& device, const  Tensor<SOURCE_SPEC>& source, Tensor<TARGET_SPEC>& target, const typename SOURCE_SPEC::T polyak) {
        using T = typename SOURCE_SPEC::T;
        tensor::Operation<binary_kernels::update<T>, binary_kernels::PolyakParameters<T>> params{};
        params.parameter.polyak = polyak;
        binary_operation(device, params, source, target);
    }

    template<typename DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void update_squared(DEVICE& device, const  Tensor<SOURCE_SPEC>& source, Tensor<TARGET_SPEC>& target, const typename SOURCE_SPEC::T polyak) {
        using T = typename SOURCE_SPEC::T;
        tensor::Operation<binary_kernels::update_squared<T>, binary_kernels::PolyakParameters<T>> params{};
        params.parameter.polyak = polyak;
        binary_operation(device, params, source, target);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif
#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_OPTIMIZERS_SGD_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_OPTIMIZERS_SGD_H

#include "../../../nn/parameters/parameters.h"
#include "../../../utils/generic/typing.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::optimizers{
    namespace sgd{
        template<typename T_T>
        struct Parameters{
            using T = T_T;
            T learning_rate;
            T momentum;
            T weight_decay;
            bool nesterov;
        };
        template <typename T_TYPE_POLICY>
        struct DefaultParameters{
            using T = typename T_TYPE_POLICY::DEFAULT;
            static constexpr T LEARNING_RATE = 0.1;
            static constexpr T MOMENTUM = 0.9;
            static constexpr T WEIGHT_DECAY = 0;
            static constexpr bool NESTEROV = false;
            static constexpr bool ENABLE_WEIGHT_DECAY = false;
        };
        template <typename T_TYPE_POLICY, typename T_TI, typename T_DEFAULT_PARAMETERS = DefaultParameters<T_TYPE_POLICY>, bool T_DYNAMIC_ALLOCATION = true>
        struct Specification{
            using T = typename T_TYPE_POLICY::DEFAULT;
            using TI = T_TI;
            using DEFAULT_PARAMETERS = T_DEFAULT_PARAMETERS;
            static constexpr bool ENABLE_WEIGHT_DECAY = DEFAULT_PARAMETERS::ENABLE_WEIGHT_DECAY;
            static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
        };
    }
    template<typename T_SPEC>
    struct SGD{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using DEFAULT_PARAMETERS = typename SPEC::DEFAULT_PARAMETERS;
        using PARAMETERS = sgd::Parameters<T>;
        Tensor<tensor::Specification<PARAMETERS, TI, tensor::Shape<TI, 1>, SPEC::DYNAMIC_ALLOCATION>> parameters;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::parameters{
    struct SGD{
        template <typename T_TYPE_POLICY, typename T_TI, typename T_SHAPE, typename T_GROUP_TAG, typename T_CATEGORY_TAG, bool T_DYNAMIC_ALLOCATION, bool T_CONST=false, typename T_NUMERIC_CATEGORY = numeric_types::categories::Parameter>
        struct Specification{
            using TYPE_POLICY = T_TYPE_POLICY;
            using TI = T_TI;
            using SHAPE = T_SHAPE;
            using GROUP_TAG = T_GROUP_TAG;
            using CATEGORY_TAG = T_CATEGORY_TAG;
            static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
            static constexpr bool CONST = T_CONST;
            using NUMERIC_CATEGORY = T_NUMERIC_CATEGORY;
        };
        template <typename T_SPEC, bool T_USE_MASTER_PARAMETERS = T_SPEC::TYPE_POLICY::template IS_SET<numeric_types::categories::MasterParameter> && !utils::typing::is_same_v<typename Gradient::Instance<T_SPEC>::PARENT::T_PARAMETER, typename T_SPEC::TYPE_POLICY::template GET<numeric_types::categories::MasterParameter>>>
        struct Instance;
        template <typename T_SPEC>
        struct Instance<T_SPEC, false>: Gradient::Instance<T_SPEC>{
            using SPEC = T_SPEC;
            using T_PARAMETER = typename Gradient::Instance<T_SPEC>::PARENT::T_PARAMETER;
            using T_MASTER_PARAMETER = typename T_SPEC::TYPE_POLICY::template GET<numeric_types::categories::MasterParameter>;
            static constexpr bool USE_MASTER_PARAMETERS = false;
            using T_VELOCITY = typename T_SPEC::TYPE_POLICY::template GET<numeric_types::categories::OptimizerState>;
            using TENSOR_SPEC = tensor::Specification<T_VELOCITY, typename SPEC::TI, typename SPEC::SHAPE, SPEC::DYNAMIC_ALLOCATION, tensor::RowMajorStride<typename SPEC::SHAPE>, SPEC::CONST>;
            Tensor<TENSOR_SPEC> velocity;
        };
        template <typename T_SPEC>
        struct Instance<T_SPEC, true>: Gradient::Instance<T_SPEC>{
            using SPEC = T_SPEC;
            using T_PARAMETER = typename Gradient::Instance<T_SPEC>::PARENT::T_PARAMETER;
            using T_MASTER_PARAMETER = typename T_SPEC::TYPE_POLICY::template GET<numeric_types::categories::MasterParameter>;
            static constexpr bool USE_MASTER_PARAMETERS = true;
            using T_VELOCITY = typename T_SPEC::TYPE_POLICY::template GET<numeric_types::categories::OptimizerState>;
            using TENSOR_SPEC = tensor::Specification<T_VELOCITY, typename SPEC::TI, typename SPEC::SHAPE, SPEC::DYNAMIC_ALLOCATION, tensor::RowMajorStride<typename SPEC::SHAPE>, SPEC::CONST>;
            Tensor<TENSOR_SPEC> velocity;
            using MASTER_TENSOR_SPEC = tensor::Specification<T_MASTER_PARAMETER, typename SPEC::TI, typename SPEC::SHAPE, SPEC::DYNAMIC_ALLOCATION, tensor::RowMajorStride<typename SPEC::SHAPE>, SPEC::CONST>;
            Tensor<MASTER_TENSOR_SPEC> master_parameters;
        };
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif

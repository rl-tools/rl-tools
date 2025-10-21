#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NUMERIC_TYPES_PERSIST_CODE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NUMERIC_TYPES_PERSIST_CODE_H
#include "../rl_tools.h"
#include "policy.h"
#include "../containers/matrix/persist_code.h"
#include "../persist/code.h"
#include "../nn/numeric_types.h"
#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    inline std::string get_type_string(nn::numeric_type_categories::Parameter){
        return "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::numeric_type_categories::Parameter";
    }
    inline std::string get_type_string(nn::numeric_type_categories::Accumulator){
        return "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::numeric_type_categories::Accumulator";
    }
    inline std::string get_type_string(nn::numeric_type_categories::Gradient){
        return "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::numeric_type_categories::Gradient";
    }
    inline std::string get_type_string(nn::numeric_type_categories::OptimizerState){
        return "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::numeric_type_categories::OptimizerState";
    }
    inline std::string get_type_string(nn::numeric_type_categories::Activation){
        return "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::numeric_type_categories::Activation";
    }
    inline std::string get_type_string(nn::numeric_type_categories::Buffer){
        return "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::numeric_type_categories::Buffer";
    }
    inline std::string get_type_string(nn::numeric_type_categories::Input){
        return "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::numeric_type_categories::Input";
    }
    template<typename DEFAULT, typename... USE_CASES>
    std::string to_string(numeric_types::Policy<DEFAULT, USE_CASES...>){
        std::string s = "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::numeric_types::Policy<";
        s += containers::persist::get_type_string<DEFAULT>();
        ((s += std::string(", RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::numeric_types::UseCase<")
                + get_type_string(typename USE_CASES::TAG{})
                + ", "
                + containers::persist::get_type_string<typename USE_CASES::TYPE>()
                + ">"), ...);
        s += ">";
        return s;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif

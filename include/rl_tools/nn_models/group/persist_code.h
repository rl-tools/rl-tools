#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_GROUP_PERSIST_CODE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_GROUP_PERSIST_CODE_H

#include "model.h"
#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::group{
    template<typename CHILD, typename PARENT>
    struct RelativeGroups;
    template<typename... TAGS>
    struct RelativeGroups<Membership<TAGS...>, Membership<>>{
        using TYPE = Membership<TAGS...>;
    };
    template<typename CHILD>
    struct RelativeGroups<CHILD, Membership<Default>>{
        using TYPE = CHILD;
    };
    template<>
    struct RelativeGroups<Membership<Default>, Membership<Default>>{
        using TYPE = Membership<>;
    };
    template<typename TAG, typename... CHILD, typename... PARENT>
    struct RelativeGroups<Membership<TAG, CHILD...>, Membership<TAG, PARENT...>>: RelativeGroups<Membership<CHILD...>, Membership<PARENT...>>{};

    template<typename TAG>
    struct CodeName;
    template<>
    struct CodeName<Default>{
        static constexpr const char* VALUE = "RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::group::Default";
        static constexpr const char* DEFINITION = "";
    };

    template<typename GROUPS>
    struct CodeGroups;
    template<>
    struct CodeGroups<Membership<>>{
        static std::string names(){
            return "";
        }
        static std::string parameter_type(const std::string& type){
            return type;
        }
        static std::string wrap(const std::string& content){
            return content;
        }
        static std::string headers(){
            return "";
        }
    };
    template<typename TAG, typename... REST>
    struct CodeGroups<Membership<TAG, REST...>>{
        static std::string names(){
            std::string value = CodeName<TAG>::VALUE;
            if constexpr(sizeof...(REST) > 0){
                value += ", " + CodeGroups<Membership<REST...>>::names();
            }
            return value;
        }
        static std::string parameter_type(const std::string& type){
            return "RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::group::Parameters<" + type + ", RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::group::Membership<" + names() + ">>";
        }
        static std::string wrap(const std::string& content){
            return std::string("RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn_models::Group<") + CodeName<TAG>::VALUE + ", " + CodeGroups<Membership<REST...>>::wrap(content) + ">";
        }
        static std::string headers(){
            return std::string("#include <rl_tools/nn_models/group/model.h>\n") + CodeName<TAG>::DEFINITION + "\n" + CodeGroups<Membership<REST...>>::headers();
        }
    };
    template<>
    struct CodeGroups<Membership<Default>>: CodeGroups<Membership<>>{};
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif

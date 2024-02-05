#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UI_SERVER_CLIENT_CLIENT_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UI_SERVER_CLIENT_CLIENT_H
#include <queue>
RL_TOOLS_NAMESPACE_WRAPPER_START

namespace rl_tools::ui_server::client{

    template<typename T_ENVIRONMENT>
    struct UIBuffered{
        std::string ns = "";
        std::queue<std::string> buffer;
    };

}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
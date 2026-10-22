#include <cstdio>

#ifdef HYPERLUX_CAMERA_CONFIG_HEADER
#include HYPERLUX_CAMERA_CONFIG_HEADER
#endif

#ifndef HYPERLUX_TOY_VALUE
#error "HYPERLUX_TOY_VALUE must be defined"
#endif

extern "C" {
    int hyperlux_toy_value(){
        return HYPERLUX_TOY_VALUE;
    }
    const char* hyperlux_toy_config_string(){
        static char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "value=%d", (int)HYPERLUX_TOY_VALUE);
        return buffer;
    }
    int hyperlux_toy_iface_version(){
        return 1;
    }
}

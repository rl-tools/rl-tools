#include <cstdio>

#ifndef HYPERDRONE_TOY_VALUE
#error "HYPERDRONE_TOY_VALUE must be defined"
#endif

extern "C" {
    int hyperdrone_toy_value(){
        return HYPERDRONE_TOY_VALUE;
    }
    const char* hyperdrone_toy_config_string(){
        static char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "value=%d", (int)HYPERDRONE_TOY_VALUE);
        return buffer;
    }
    int hyperdrone_toy_iface_version(){
        return 1;
    }
}

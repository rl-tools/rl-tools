#include <wasi/api.h>

// Assimp profiling may read the clock; entropy must never enter the reference renderer.
extern "C" __wasi_errno_t __wrap___wasi_clock_time_get(__wasi_clockid_t, __wasi_timestamp_t, __wasi_timestamp_t* time){
    *time = 0;
    return 0;
}

extern "C" __wasi_errno_t __wrap___wasi_random_get(uint8_t*, size_t){
    __builtin_trap();
}

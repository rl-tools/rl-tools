# Migration from the pre-auto-configure flag surface: features are enabled by default now and
# opted out via RL_TOOLS_*_DISABLE_* options.
# - A legacy cache entry that agrees with the new default behavior is scrubbed with a notice.
# - A BOOL-typed OFF entry whose value is just the old option() default is a stale leftover of a
#   previous configure (option() types passed-in entries), not an explicit choice: scrubbed too,
#   so existing build trees keep reconfiguring under the new defaults.
# - Anything else would silently change behavior: fail with the replacement flag.
function(rl_tools_legacy_flag LEGACY_NAME OLD_DEFAULT REPLACEMENT_HINT)
    if(DEFINED CACHE{${LEGACY_NAME}})
        if(${LEGACY_NAME})
            message(STATUS "RLtools: legacy flag ${LEGACY_NAME} matches the default behavior now, removing it from the cache")
            unset(${LEGACY_NAME} CACHE)
        else()
            get_property(RL_TOOLS_LEGACY_TYPE CACHE ${LEGACY_NAME} PROPERTY TYPE)
            if(RL_TOOLS_LEGACY_TYPE STREQUAL "BOOL" AND OLD_DEFAULT STREQUAL "OFF")
                message(STATUS "RLtools: removing stale legacy cache entry ${LEGACY_NAME} (${REPLACEMENT_HINT})")
                unset(${LEGACY_NAME} CACHE)
            else()
                message(FATAL_ERROR "RLtools: ${LEGACY_NAME} has been replaced: ${REPLACEMENT_HINT}")
            endif()
        endif()
    endif()
endfunction()

rl_tools_legacy_flag(RL_TOOLS_ENABLE_TARGETS ON "targets are enabled by default, use RL_TOOLS_DISABLE_TARGETS=ON to disable them")
rl_tools_legacy_flag(RL_TOOLS_ENABLE_TESTS OFF "tests are enabled by default, use RL_TOOLS_DISABLE_TESTS=ON to disable them")
rl_tools_legacy_flag(RL_TOOLS_ENABLE_GIT_DIFF ON "git diff embedding is enabled by default, use RL_TOOLS_DISABLE_GIT_DIFF=ON to disable it")
rl_tools_legacy_flag(RL_TOOLS_EXPERIMENTAL "" "experimental targets are enabled by default, use RL_TOOLS_DISABLE_EXPERIMENTAL=ON to disable them")
rl_tools_legacy_flag(RL_TOOLS_ENABLE_TAR "" "tar support is enabled by default, use RL_TOOLS_DISABLE_TAR=ON to disable it")
rl_tools_legacy_flag(RL_TOOLS_NUMERIC_TYPES_ENABLE_BF16 "" "bf16 support is auto-detected, use RL_TOOLS_NUMERIC_TYPES_DISABLE_BF16=ON to disable it")
rl_tools_legacy_flag(RL_TOOLS_TESTS_ENABLE_EIGEN "" "Eigen is auto-detected, use RL_TOOLS_TESTS_DISABLE_EIGEN=ON to disable it")
rl_tools_legacy_flag(RL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO OFF "MuJoCo is auto-enabled, use RL_TOOLS_RL_ENVIRONMENTS_DISABLE_MUJOCO=ON to disable it")
rl_tools_legacy_flag(RL_TOOLS_RENDERING_ENABLE_RAYTRACING OFF "raytracing is auto-enabled, use RL_TOOLS_RENDERING_DISABLE_RAYTRACING=ON to disable it")

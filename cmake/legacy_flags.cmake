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
rl_tools_legacy_flag(RL_TOOLS_RENDERING_ENABLE_RAYTRACING OFF "HyperLux is auto-enabled, use RL_TOOLS_DISABLE_HYPERLUX=ON to disable it")

# A renamed flag keeps its semantics under the new name. A stale entry at the default is scrubbed;
# so is one that a configure already overrides under the new name (the JIT build trees FORCE it);
# any other value fails with the new name, since the old one would otherwise be ignored silently.
function(rl_tools_renamed_flag OLD_NAME NEW_NAME DEFAULT)
    if(DEFINED CACHE{${OLD_NAME}})
        if("$CACHE{${OLD_NAME}}" STREQUAL "${DEFAULT}")
            message(STATUS "RLtools: removing stale cache entry ${OLD_NAME} (renamed to ${NEW_NAME})")
            unset(${OLD_NAME} CACHE)
        elseif(DEFINED CACHE{${NEW_NAME}} AND NOT "$CACHE{${NEW_NAME}}" STREQUAL "${DEFAULT}")
            message(STATUS "RLtools: removing cache entry ${OLD_NAME}, superseded by ${NEW_NAME}=$CACHE{${NEW_NAME}}")
            unset(${OLD_NAME} CACHE)
        else()
            message(FATAL_ERROR "RLtools: ${OLD_NAME} has been renamed to ${NEW_NAME}; pass -D${NEW_NAME}=$CACHE{${OLD_NAME}} and drop the old entry with cmake -U ${OLD_NAME}")
        endif()
    endif()
endfunction()

rl_tools_renamed_flag(RL_TOOLS_RENDERING_DISABLE_RAYTRACING RL_TOOLS_DISABLE_HYPERLUX OFF)
rl_tools_renamed_flag(RL_TOOLS_RENDERING_RAYTRACING_BACKEND RL_TOOLS_HYPERLUX_BACKEND AUTO)
rl_tools_renamed_flag(RL_TOOLS_RENDERING_RAYTRACING_DISABLE_SPY RL_TOOLS_HYPERLUX_DISABLE_SPY OFF)
rl_tools_renamed_flag(RL_TOOLS_RENDERING_RAYTRACING_DISABLE_OPTIX RL_TOOLS_HYPERLUX_DISABLE_OPTIX OFF)
rl_tools_renamed_flag(RL_TOOLS_RENDERING_RAYTRACING_DISABLE_METAL RL_TOOLS_HYPERLUX_DISABLE_METAL OFF)
rl_tools_renamed_flag(RL_TOOLS_RENDERING_RAYTRACING_DISABLE_VULKAN RL_TOOLS_HYPERLUX_DISABLE_VULKAN OFF)
rl_tools_renamed_flag(RL_TOOLS_RENDERING_RAYTRACING_DISABLE_WEBGPU RL_TOOLS_HYPERLUX_DISABLE_WEBGPU OFF)
rl_tools_renamed_flag(RL_TOOLS_RENDERING_RAYTRACING_ENABLE_WASM_GOLDENS RL_TOOLS_HYPERLUX_ENABLE_WASM_GOLDENS OFF)
rl_tools_renamed_flag(RL_TOOLS_REQUIRE_RAYTRACING_GOLDENS RL_TOOLS_REQUIRE_HYPERLUX_GOLDENS OFF)
rl_tools_renamed_flag(RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS RL_TOOLS_HYPERLUX_SIM_BENCHMARK_NUM_ENVS 4096)

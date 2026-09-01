if(NOT RL_TOOLS_NUMERIC_TYPES_DISABLE_BF16)
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles("
        int main(){
            __bf16 value = static_cast<__bf16>(1.0f);
            static_assert(sizeof(__bf16) == 2, \"bf16 size\");
            return static_cast<int>(static_cast<float>(value));
        }
    " RL_TOOLS_BF16_SUPPORTED)
    if(RL_TOOLS_BF16_SUPPORTED)
        # CXX-only: .cu device compiles use __nv_bfloat16 through the type policy instead
        target_compile_definitions(rl_tools_full INTERFACE $<$<COMPILE_LANGUAGE:CXX>:RL_TOOLS_NUMERIC_TYPES_ENABLE_BF16>)
        set(RL_TOOLS_NUMERIC_TYPES_ENABLE_BF16 ON)
        set(RL_TOOLS_SUMMARY_BF16_REASON "compiler supports __bf16")
    else()
        message(STATUS "bf16 disabled (compiler does not support __bf16)")
        set(RL_TOOLS_SUMMARY_BF16_REASON "compiler does not support __bf16")
    endif()
endif()

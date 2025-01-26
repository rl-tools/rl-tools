#include "../rl_tools.h"
#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NUMERIC_TYPES_BF16_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NUMERIC_TYPES_BF16_H

#include <cstdint>
#include <cstring>


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace numeric_types {
        struct bfloat16{
            std::uint16_t value;
            bfloat16(float f){
                std::uint32_t bits;
                std::memcpy(&bits, &f, sizeof(bits));
                std::uint16_t top = static_cast<std::uint16_t>(bits >> 16);
                std::uint16_t lower = static_cast<std::uint16_t>(bits & 0xFFFF);

                if (lower > 0x8000 || (lower == 0x8000 && (top & 1) != 0)) {
                    top++;
                }
                this->value = top;
            }
            operator float() const {
                std::uint32_t bits = static_cast<std::uint32_t>(this->value) << 16;
                float f;
                std::memcpy(&f, &bits, sizeof(f));
                return f;
            };
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


::rl_tools::numeric_types::bfloat16 operator+(RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 a, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 b) {
    return RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16(static_cast<float>(a) + static_cast<float>(b));
}

RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 operator-(RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 a, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 b) {
    return RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16(static_cast<float>(a) - static_cast<float>(b));
}

RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 operator*(RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 a, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 b) {
    return RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16(static_cast<float>(a) * static_cast<float>(b));
}

RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 operator/(RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 a, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 b) {
    return RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16(static_cast<float>(a) / static_cast<float>(b));
}

bool operator==(RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 a, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 b) { return a.value == b.value; }
bool operator!=(RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 a, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 b) { return a.value != b.value; }

bool operator>(RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 a, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 b) { return static_cast<float>(a) > static_cast<float>(b); }
bool operator<(RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 a, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 b) { return static_cast<float>(a) < static_cast<float>(b); }
bool operator>=(RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 a, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 b) { return static_cast<float>(a) >= static_cast<float>(b); }
bool operator<=(RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 a, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 b) { return static_cast<float>(a) <= static_cast<float>(b); }

RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16& operator+=(RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16& a, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 b) { return a = a + b; }
RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16& operator-=(RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16& a, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 b) { return a = a - b; }
RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16& operator*=(RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16& a, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 b) { return a = a * b; }
RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16& operator/=(RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16& a, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::numeric_types::bfloat16 b) { return a = a / b; }

#endif
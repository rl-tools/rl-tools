#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_CONTAINERS_PERSIST_CODE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_CONTAINERS_PERSIST_CODE_H

#include "../persist/code.h"
#include "../utils/generic/typing.h"
#include <sstream>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace containers::persist{
        using STORAGE_TYPE = unsigned char;
        static_assert(sizeof(STORAGE_TYPE) == 1);
        constexpr auto INDEX_TYPE = "unsigned int";
        template <typename T>
        auto get_type_string(){
            static_assert(utils::typing::is_same_v<T, float> || utils::typing::is_same_v<T, double> || utils::typing::is_same_v<T, int> || utils::typing::is_same_v<T, unsigned int> || utils::typing::is_same_v<T, long> || utils::typing::is_same_v<T, unsigned long> || utils::typing::is_same_v<T, long long> || utils::typing::is_same_v<T, unsigned long long> || utils::typing::is_same_v<T, char> || utils::typing::is_same_v<T, unsigned char> || utils::typing::is_same_v<T, short> || utils::typing::is_same_v<T, unsigned short>);
            if constexpr(std::is_same_v<T, float>){
                return "float";
            } else if constexpr(std::is_same_v<T, double>){
                return "double";
            } else if constexpr(std::is_same_v<T, int>){
                return "int";
            } else if constexpr(std::is_same_v<T, unsigned int>){
                return "unsigned int";
            } else if constexpr(std::is_same_v<T, long>){
                return "long";
            } else if constexpr(std::is_same_v<T, unsigned long>){
                return "unsigned long";
            } else if constexpr(std::is_same_v<T, long long>){
                return "long long";
            } else if constexpr(std::is_same_v<T, unsigned long long>){
                return "unsigned long long";
            } else if constexpr(std::is_same_v<T, char>){
                return "char";
            } else if constexpr(std::is_same_v<T, unsigned char>){
                return "unsigned char";
            } else if constexpr(std::is_same_v<T, short>){
                return "short";
            } else if constexpr(std::is_same_v<T, unsigned short>){
                return "unsigned short";
            } else {
                return "Unsupported type";
            }
        }
    }
    template<typename DEVICE, typename SPEC>
    persist::Code save_split(DEVICE& device, Matrix<SPEC>& m, std::string name, bool const_declaration=false, typename DEVICE::index_t indent=0){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        static_assert(utils::typing::is_same_v<containers::persist::STORAGE_TYPE, unsigned char>);
        static_assert(sizeof(T) % sizeof(containers::persist::STORAGE_TYPE) == 0);
        std::stringstream indent_ss;
        for(TI i=0; i < indent; i++){
            indent_ss << "    ";
        }
        std::string ind = indent_ss.str();
        std::stringstream ss_header;
        ss_header << "#include <rl_tools/containers.h>\n";
        std::stringstream ss;
        ss << ind << "namespace " << name << " {\n";
        ss << ind << "    static_assert(sizeof(" << containers::persist::get_type_string<containers::persist::STORAGE_TYPE>() << ") == 1);\n";
        ss << ind << "    alignas(" << containers::persist::get_type_string<T>() << ") " << (const_declaration ? "const " : "") << containers::persist::get_type_string<containers::persist::STORAGE_TYPE>() << " memory[] = {";
        bool first = true;
        for(TI i=0; i < SPEC::ROWS; i++){
            for(TI j=0; j < SPEC::COLS; j++){
                auto value = get(m, i, j);
                auto* ptr = reinterpret_cast<containers::persist::STORAGE_TYPE*>(&value);
                for(int k=0; k < sizeof(T); k++){
                    if(!first){
                        ss << ", ";
                    }
                    ss << (int)ptr[k];
                    first = false;
                }
            }
        }
        ss << "};\n";
        ss << ind << "    using CONTAINER_SPEC = RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::matrix::Specification<" << containers::persist::get_type_string<T>() << ", " << containers::persist::get_type_string<TI>() << ", " << SPEC::ROWS << ", " << SPEC::COLS << ", " << "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::matrix::layouts::RowMajorAlignment<" << containers::persist::get_type_string<TI>() << ", " << 1 << ">>;\n";
        ss << ind << "    using CONTAINER_TYPE = RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::MatrixDynamic<CONTAINER_SPEC>;\n";
        ss << ind << "    " << (const_declaration ? "const " : "") << "CONTAINER_TYPE container = {(" << containers::persist::get_type_string<T>() << "*)" << "memory}; \n";
        ss << ind << "}\n";
        return {ss_header.str(), ss.str()};
    }
    template<typename DEVICE, typename SPEC>
    std::string save_code(DEVICE& device, Matrix<SPEC>& m, std::string name, bool const_declaration, typename DEVICE::index_t indent=0){
        auto code = save_split(device, m, name, const_declaration, indent);
        return code.header + code.body;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif

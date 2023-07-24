#ifndef BACKPROP_TOOLS_PERSIST_CODE_H
#define BACKPROP_TOOLS_PERSIST_CODE_H

namespace backprop_tools::persist{
    struct Code{
        std::string header;
        std::string body;
    };
}

namespace backprop_tools{
    template <typename DEVICE>
    persist::Code embed_in_namespace(DEVICE&, persist::Code c, std::string name, typename DEVICE::index_t indent = 0){
        using TI = typename DEVICE::index_t;
        std::stringstream indent_ss;
        for(TI i=0; i < indent; i++){
            indent_ss << "    ";
        }
        std::string ind = indent_ss.str();
        std::stringstream ss_header;
        std::stringstream ss;
        ss_header << c.header;
        ss << ind << "namespace " << name << " {\n";
        ss << c.body;
        ss << ind << "}\n";
        return {ss_header.str(), ss.str()};
    }
}

#endif
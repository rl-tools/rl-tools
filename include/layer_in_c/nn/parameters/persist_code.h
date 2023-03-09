
#include <layer_in_c/containers/persist_code.h>
#include "parameters.h"
#include <sstream>
namespace layer_in_c {
    std::string get_type_string(nn::parameters::Plain p){
        return "layer_in_c::nn::parameters::Plain";
    }
    std::string get_type_string(nn::parameters::Gradient p){
        return "layer_in_c::nn::parameters::Gradient";
    }
    template<typename DEVICE, typename CONTAINER>
    persist::Code save_split(DEVICE &device, nn::parameters::Plain::instance<CONTAINER>& parameter, std::string name, bool const_declaration=false, typename DEVICE::index_t indent=0){
        using TI = typename DEVICE::index_t;
        std::stringstream indent_ss;
        for(TI i=0; i < indent; i++){
            indent_ss << "    ";
        }
        std::string ind = indent_ss.str();
        std::stringstream ss, ss_header;
        ss << ind << "namespace " << name << " {\n";
        auto container = save_split(device, parameter.parameters, "container", const_declaration, indent+1);
        ss_header << container.header;
        ss << container.body;
        ss << ind << "    " << (const_declaration ? "const " : "") << "layer_in_c::nn::parameters::Plain::instance<container::CONTAINER_TYPE> parameters = {container::container};\n";
        ss << ind << "}\n";
        return {"", ss.str()};
    }
}


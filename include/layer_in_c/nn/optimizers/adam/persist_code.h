#ifndef LAYER_IN_C_NN_OPTIMIZERS_ADAM_PERSIST_CODE_H
#define LAYER_IN_C_NN_OPTIMIZERS_ADAM_PERSIST_CODE_H

#include "adam.h"

#include <string>
namespace layer_in_c{
    std::string get_type_string(nn::parameters::Adam p){
        return "layer_in_c::nn::parameters::Adam";
    }
}


#endif

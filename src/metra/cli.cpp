#include <metra/metra.h>

#include <iostream>
#include <string>

int main(int argc, char** argv){
    if(argc < 3){
        std::cerr << "Usage: " << argv[0] << " <name> <value-json> [more values ...]" << std::endl;
        std::cerr << "Logs metric values (JSON scalars, lists, or structs) to the metra server, one row per value." << std::endl;
        std::cerr << "Environment: METRA_URL (server base URL, required), METRA_COMMIT (overrides commit detection), METRA_RUN (overrides the generated run id)." << std::endl;
        return 1;
    }
    metra::Config config = metra::config_from_environment();
    if(config.url.empty()){
        std::cerr << "metra: METRA_URL not set" << std::endl;
        return 1;
    }
    std::string name = argv[1];
    for(int argument_i = 2; argument_i < argc; argument_i++){
        std::string error;
        if(!metra::log_raw(config, name, argv[argument_i], error)){
            std::cerr << error << std::endl;
            return 1;
        }
    }
    return 0;
}

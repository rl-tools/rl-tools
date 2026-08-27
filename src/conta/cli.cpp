#include <conta/conta.h>

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv){
    if(argc < 2){
        std::cerr << "Usage: " << argv[0] << " <sha1|conta:sha1> [more hashes ...]" << std::endl;
        std::cerr << "Resolves content-addressed blobs to local filesystem paths (downloading into the cache if required) and prints one path per line." << std::endl;
        std::cerr << "Environment: CONTA_ROOT (read-only store, disables downloads), CONTA_CACHE (cache directory), CONTA_URL (download base URL)." << std::endl;
        return 1;
    }
    std::vector<std::string> hashes;
    for(int argument_i = 1; argument_i < argc; argument_i++){
        std::string argument = argv[argument_i];
        if(argument.rfind("conta:", 0) == 0){
            argument = argument.substr(6);
        }
        hashes.push_back(argument);
    }
    std::vector<std::string> paths;
    std::string error;
    if(!conta::resolve(hashes, paths, error)){
        std::cerr << error << std::endl;
        return 1;
    }
    for(const std::string& path: paths){
        std::cout << path << std::endl;
    }
    return 0;
}

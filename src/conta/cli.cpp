#include <conta/conta.h>

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv){
    if(argc < 2){
        std::cerr << "Usage: " << argv[0] << " <sha1|conta:sha1> [more hashes ...]" << std::endl;
        std::cerr << "Resolves content-addressed blobs to local filesystem paths (downloading into the cache through the conta index if required) and prints one path per line." << std::endl;
        std::cerr << "Environment: CONTA_ROOT (read-only store, disables the network), CONTA_CACHE (cache directory), CONTA_URL (fetch <url>/<sha1> directly, without the index)." << std::endl;
        return 1;
    }
    std::vector<std::string> hashes(argv + 1, argv + argc);
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

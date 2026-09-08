#pragma once

namespace rl_tools{
    template<typename DEVICE>
    auto resolve_training_scenes(DEVICE& device, const char* scene_arg){
        using namespace rl::environments::l2f_visual::training;
        std::vector<std::string> scene_paths;
        if(std::filesystem::is_directory(scene_arg)){
            std::vector<std::string> all_glbs;
            for(auto& entry : std::filesystem::directory_iterator(scene_arg)){
                if(entry.path().extension() == ".glb"){
                    all_glbs.push_back(entry.path().string());
                }
            }
            std::sort(all_glbs.begin(), all_glbs.end(), [](const std::string& a, const std::string& b){
                auto extract_number = [](const std::string& path) -> int {
                    auto filename = std::filesystem::path(path).stem().string();
                    auto pos = filename.rfind('-');
                    if(pos != std::string::npos){
                        try{
                            return std::stoi(filename.substr(pos + 1));
                        }
                        catch(...){}
                    }
                    return 0;
                };
                return extract_number(a) < extract_number(b);
            });
            if(static_cast<TI>(all_glbs.size()) < N_TOTAL_SCENES){
                std::cerr << "Need at least " << N_TOTAL_SCENES << " GLB scenes, found " << all_glbs.size() << std::endl;
                utils::assert_exit(device, false, "Not enough GLB scenes");
            }
            for(TI i = 0; i < N_TOTAL_SCENES; i++){
                scene_paths.push_back(all_glbs[i]);
            }
            std::cout << "Selected " << scene_paths.size() << " scenes from " << scene_arg << std::endl;
        } else {
            scene_paths.resize(N_TOTAL_SCENES, scene_arg);
            std::cout << "Replicating single scene across " << N_TOTAL_SCENES << " renderers: " << scene_arg << std::endl;
        }

        return scene_paths;
    }
}

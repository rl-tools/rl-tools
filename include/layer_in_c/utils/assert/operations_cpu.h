

#include <cstdlib>
namespace layer_in_c::utils{
    template <typename DEV_SPEC, typename T>
    void assert_exit(devices::CPU<DEV_SPEC>& dev, bool condition, T message){
        if(!condition){
            logging::text(dev.logger, message);
            std::exit(-1);
        }
    }
}

#include <layer_in_c/devices.h>

#include <cstdlib>
namespace layer_in_c::utils{
    template <typename DEV_SPEC, typename T>
    void assert_exit(const devices::CPU<DEV_SPEC>& dev, bool condition, T message){
        if(!condition){
            logging::text(typename DEV_SPEC::LOGGING(), message);
            std::exit(-1);
        }
    }
}

#include <cstdlib>
namespace layer_in_c::utils{
    template <typename T>
    void assert_exit(bool condition, T message){
        if(!condition){
            logging::text(message);
            std::exit(-1);
        }
    }
}

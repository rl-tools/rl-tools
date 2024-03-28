#include <rl_tools/operations/arm.h>
namespace rlt = rl_tools;
using DEVICE = rlt::devices::DefaultARM;
using T = float;
using TI = typename DEVICE::index_t;


int main(){
    DEVICE device;

    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 10, 10>> A, B, C;
//    rlt::malloc(device, A);
//    rlt::malloc(device, B);
//    rlt::malloc(device, C);
    rlt::set_all(device, A, 1);
    rlt::set_all(device, B, 2);
    rlt::multiply(device, A, B, C);
    return rlt::get(C, 0, 0);
}
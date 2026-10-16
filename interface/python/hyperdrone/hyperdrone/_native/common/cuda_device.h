#pragma once
#include <cuda_runtime.h>

namespace hyperdrone {
    struct CurrentCudaDevice {
        int previous = 0;
        int selected;

        explicit CurrentCudaDevice(int device): selected(device){
            cudaGetDevice(&previous);
            if(previous != selected){
                cudaSetDevice(selected);
            }
        }
        ~CurrentCudaDevice(){
            if(previous != selected){
                cudaSetDevice(previous);
            }
        }
    };
}

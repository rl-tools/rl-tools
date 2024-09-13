template <typename DEVICE, typename SPEC, std::enable_if_t<rlt::length(typename SPEC::SHAPE{}) == 1, int> = 0>
void fill(DEVICE& device, rlt::Tensor<SPEC>& tensor) {
    using T = typename SPEC::T;
    using TI = typename DEVICE::index_t;
    for(TI i=0; i < rlt::get<0>(typename SPEC::SHAPE{}); i++){
        rlt::set(device, tensor, i, i);
    }
}

template <typename DEVICE, typename SPEC, std::enable_if_t<rlt::length(typename SPEC::SHAPE{}) == 2, int> = 0>
void fill(DEVICE& device, rlt::Tensor<SPEC>& tensor) {
    using T = typename SPEC::T;
    using TI = typename DEVICE::index_t;
    for(TI i=0; i < rlt::get<0>(typename SPEC::SHAPE{}); i++){
        for(TI j=0; j < rlt::get<1>(typename SPEC::SHAPE{}); j++){
            rlt::set(device, tensor, i+j, i, j);
        }
    }
}

template <typename DEVICE, typename SPEC, std::enable_if_t<rlt::length(typename SPEC::SHAPE{}) == 3, int> = 0>
void fill(DEVICE& device, rlt::Tensor<SPEC>& tensor) {
    using T = typename SPEC::T;
    using TI = typename DEVICE::index_t;
    for(TI i=0; i < rlt::get<0>(typename SPEC::SHAPE{}); i++){
        for(TI j=0; j < rlt::get<1>(typename SPEC::SHAPE{}); j++){
            for(TI k=0; k < rlt::get<2>(typename SPEC::SHAPE{}); k++){
                rlt::set(device, tensor, i+j+k, i, j, k);
            }
        }
    }
}

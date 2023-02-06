#ifndef LAYER_IN_C_NN_UTILS_POLYAK
#define LAYER_IN_C_NN_UTILS_POLYAK


namespace layer_in_c::utils::polyak {
    // todo: polyak factor as template parameter (reciprocal INT e.g.)
    template<typename DEVICE, typename TARGET_SPEC, typename SOURCE_SPEC>
    void update(DEVICE& dev, Matrix<TARGET_SPEC>& target, const Matrix<SOURCE_SPEC>& source, const typename TARGET_SPEC::T polyak) {
        static_assert(containers::check_structure<TARGET_SPEC, SOURCE_SPEC>);
        using SPEC = TARGET_SPEC;
        for(typename DEVICE::index_t i = 0; i < SPEC::ROWS; i++) {
            for(typename DEVICE::index_t j = 0; j < SPEC::COLS; j++) {
                target.data[i * SPEC::COLS + j] = polyak * target.data[i * SPEC::COLS + j] + (1 - polyak) * source.data[i * SPEC::COLS + j];
            }
        }
    }

    template<typename DEVICE, typename TARGET_SPEC, typename SOURCE_SPEC>
    void update_squared(DEVICE& dev, Matrix<TARGET_SPEC>& target, const Matrix<SOURCE_SPEC>& source, const typename TARGET_SPEC::T polyak) {
        static_assert(containers::check_structure<TARGET_SPEC, SOURCE_SPEC>);
        using SPEC = TARGET_SPEC;
        for(typename DEVICE::index_t i = 0; i < SPEC::ROWS; i++) {
            for(typename DEVICE::index_t j = 0; j < SPEC::COLS; j++) {
                typename SPEC::T s = source.data[i * SPEC::COLS + j];
                target.data[i * SPEC::COLS + j] = polyak * target.data[i * SPEC::COLS + j] + (1 - polyak) * s * s;
            }
        }
    }
}


#endif
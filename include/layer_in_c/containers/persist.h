#include <highfive/H5File.hpp>

namespace layer_in_c {
    template<typename DEVICE, typename SPEC>
    void save(DEVICE &device, Matrix<SPEC>& m, HighFive::Group group, std::string dataset_name) {
        using T = typename SPEC::T;
        if constexpr(SPEC::ROWS == 1){
            std::vector<T> data(SPEC::COLS);
            for(typename DEVICE::index_t i=0; i < SPEC::COLS; i++){
                data[i] = m.data[index(m, 0, i)];
            }
            group.createDataSet(dataset_name, data);
        }
        else{
            std::vector<std::vector<T>> data(SPEC::ROWS);
            for(typename DEVICE::index_t i=0; i < SPEC::ROWS; i++){
                data[i] = std::vector<T>(SPEC::COLS);
                for(typename DEVICE::index_t j=0; j < SPEC::COLS; j++){
                    data[i][j] = m.data[index(m, i, j)];
                }
            }
            group.createDataSet(dataset_name, data);
        }
    }

    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, Matrix<SPEC>& m, HighFive::Group group, std::string dataset_name) {
        auto dataset = group.getDataSet(dataset_name);
        auto dims = dataset.getDimensions();
        if(dims.size() == 1){
            assert(SPEC::ROWS == 1);
            assert(dims[0] == SPEC::COLS);
            std::vector<typename SPEC::T> data;
            dataset.read(data);
            for(typename DEVICE::index_t j=0; j < SPEC::COLS; j++){
                m.data[index(m, 0, j)] = data[j];
            }
        }
        else{
            assert(dims[0] == SPEC::ROWS);
            assert(dims[1] == SPEC::COLS);
            std::vector<std::vector<typename SPEC::T>> data;
            dataset.read(data);
            for(typename DEVICE::index_t i=0; i < SPEC::ROWS; i++){
                for(typename DEVICE::index_t j=0; j < SPEC::COLS; j++){
                    m.data[index(m, i, j)] = data[i][j];
                }
            }
        }
    }
}

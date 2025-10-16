#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_PERSIST_BACKENDS_HDF5_OPERATIONS_GENERIC)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_PERSIST_BACKENDS_HDF5_OPERATIONS_GENERIC

#include "hdf5.h"
#include <highfive/H5File.hpp>
#include "../../../containers/matrix/persist_hdf5.h"
#include "../../../containers/tensor/persist_hdf5.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC, typename GROUP_SPEC>
    void load(DEVICE& device, Matrix<SPEC>& matrix, const persist::backends::hdf5::Group<GROUP_SPEC>& group, std::string dataset_name) {
        load(device, matrix, group.group, dataset_name);
    }
    template<typename DEVICE, typename SPEC, typename GROUP_SPEC>
    void load(DEVICE& device, Tensor<SPEC>& tensor, const persist::backends::hdf5::Group<GROUP_SPEC>& group, std::string dataset_name) {
        load(device, tensor, group.group, dataset_name);
    }

    template<typename DEVICE, typename SPEC, typename GROUP_SPEC>
    void save(DEVICE& device, Matrix<SPEC>& m, persist::backends::hdf5::Group<GROUP_SPEC>& group, std::string dataset_name) {
        save(device, m, group.group, dataset_name);
    }
    template<typename DEVICE, typename SPEC, typename GROUP_SPEC>
    void save(DEVICE& device, Tensor<SPEC>& tensor, persist::backends::hdf5::Group<GROUP_SPEC>& group, std::string dataset_name) {
        save(device, tensor, group.group, dataset_name);
    }
    template<typename DEVICE, typename SPEC>
    persist::backends::hdf5::Group<SPEC> create_group(DEVICE& device, persist::backends::hdf5::Group<SPEC>& group, std::string name) {
        return {group.group.createGroup(name)};
    }
    template<typename DEVICE>
    persist::backends::hdf5::Group<persist::backends::hdf5::GroupSpecification<>> create_group(DEVICE& device, HighFive::File& file, std::string name) {
        return {file.createGroup(name)};
    }
    template<typename TYPE, typename DEVICE, typename SPEC>
    void set_attribute(DEVICE& device, persist::backends::hdf5::Group<SPEC>& group, std::string name, TYPE value) {
        group.group.template createAttribute<TYPE>(name, value);
    }
    template<typename DEVICE, typename SPEC>
    persist::backends::hdf5::Group<SPEC> get_group(DEVICE& device, persist::backends::hdf5::Group<SPEC>& group, std::string name) {
        return {group.group.getGroup(name)};
    }
    template<typename DEVICE>
    persist::backends::hdf5::Group<persist::backends::hdf5::GroupSpecification<>> get_group(DEVICE& device, HighFive::File& file, std::string name) {
        return {file.getGroup(name)};
    }
    template<typename DEVICE, typename SPEC>
    bool group_exists(DEVICE& device, persist::backends::hdf5::Group<SPEC>& group, std::string name) {
        return group.group.exist(name);
    }
    template<typename DEVICE, typename SPEC, typename T>
    void create_dataset(DEVICE& device, persist::backends::hdf5::Group<SPEC>& group, std::string name, const std::vector<T>& data) {
        group.group.createDataSet(name, data);
    }
    template<typename DEVICE, typename SPEC, typename T>
    void read_dataset(DEVICE& device, persist::backends::hdf5::Group<SPEC>& group, std::string name, std::vector<T>& data) {
        group.group.getDataSet(name).read(data);
    }
    template<typename TYPE, typename DEVICE, typename SPEC>
    TYPE get_attribute(DEVICE& device, persist::backends::hdf5::Group<SPEC>& group, std::string name) {
        return group.group.getAttribute(name).template read<TYPE>();
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
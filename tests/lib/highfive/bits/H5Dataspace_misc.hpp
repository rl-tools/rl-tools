/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5DATASPACE_MISC_HPP
#define H5DATASPACE_MISC_HPP

#include <array>
#include <initializer_list>
#include <vector>
#include <numeric>

#include <H5Spublic.h>

#include "H5Utils.hpp"
#include "H5Converter_misc.hpp"

namespace HighFive {

inline DataSpace::DataSpace(const std::vector<size_t>& dims)
    : DataSpace(dims.begin(), dims.end()) {}

template <size_t N>
inline DataSpace::DataSpace(const std::array<size_t, N>& dims)
    : DataSpace(dims.begin(), dims.end()) {}

inline DataSpace::DataSpace(const std::initializer_list<size_t>& items)
    : DataSpace(std::vector<size_t>(items)) {}

template <typename... Args>
inline DataSpace::DataSpace(size_t dim1, Args... dims)
    : DataSpace(std::vector<size_t>{dim1, static_cast<size_t>(dims)...}) {}

template <class IT, typename>
inline DataSpace::DataSpace(const IT begin, const IT end) {
    std::vector<hsize_t> real_dims(begin, end);

    if ((_hid = H5Screate_simple(int(real_dims.size()), real_dims.data(), NULL)) < 0) {
        throw DataSpaceException("Impossible to create dataspace");
    }
}

inline DataSpace::DataSpace(const std::vector<size_t>& dims, const std::vector<size_t>& maxdims) {
    if (dims.size() != maxdims.size()) {
        throw DataSpaceException("dims and maxdims must be the same length.");
    }

    std::vector<hsize_t> real_dims(dims.begin(), dims.end());
    std::vector<hsize_t> real_maxdims(maxdims.begin(), maxdims.end());

    // Replace unlimited flag with actual HDF one
    std::replace(real_maxdims.begin(),
                 real_maxdims.end(),
                 static_cast<hsize_t>(DataSpace::UNLIMITED),
                 H5S_UNLIMITED);

    if ((_hid = H5Screate_simple(int(dims.size()), real_dims.data(), real_maxdims.data())) < 0) {
        throw DataSpaceException("Impossible to create dataspace");
    }
}  // namespace HighFive

inline DataSpace::DataSpace(DataSpace::DataspaceType dtype) {
    H5S_class_t h5_dataspace_type;
    switch (dtype) {
    case DataSpace::dataspace_scalar:
        h5_dataspace_type = H5S_SCALAR;
        break;
    case DataSpace::dataspace_null:
        h5_dataspace_type = H5S_NULL;
        break;
    default:
        throw DataSpaceException(
            "Invalid dataspace type: should be "
            "dataspace_scalar or dataspace_null");
    }

    if ((_hid = H5Screate(h5_dataspace_type)) < 0) {
        throw DataSpaceException("Unable to create dataspace");
    }
}

inline DataSpace DataSpace::clone() const {
    DataSpace res;
    if ((res._hid = H5Scopy(_hid)) < 0) {
        throw DataSpaceException("Unable to copy dataspace");
    }
    return res;
}

inline size_t DataSpace::getNumberDimensions() const {
    const int ndim = H5Sget_simple_extent_ndims(_hid);
    if (ndim < 0) {
        HDF5ErrMapper::ToException<DataSetException>(
            "Unable to get dataspace number of dimensions");
    }
    return size_t(ndim);
}

inline std::vector<size_t> DataSpace::getDimensions() const {
    std::vector<hsize_t> dims(getNumberDimensions());
    if (!dims.empty()) {
        if (H5Sget_simple_extent_dims(_hid, dims.data(), NULL) < 0) {
            HDF5ErrMapper::ToException<DataSetException>("Unable to get dataspace dimensions");
        }
    }
    return details::to_vector_size_t(std::move(dims));
}

inline size_t DataSpace::getElementCount() const {
    const std::vector<size_t>& dims = getDimensions();
    return std::accumulate(dims.begin(), dims.end(), size_t{1u}, std::multiplies<size_t>());
}

inline std::vector<size_t> DataSpace::getMaxDimensions() const {
    std::vector<hsize_t> maxdims(getNumberDimensions());
    if (H5Sget_simple_extent_dims(_hid, NULL, maxdims.data()) < 0) {
        HDF5ErrMapper::ToException<DataSetException>("Unable to get dataspace dimensions");
    }

    std::replace(maxdims.begin(),
                 maxdims.end(),
                 H5S_UNLIMITED,
                 static_cast<hsize_t>(DataSpace::UNLIMITED));
    return details::to_vector_size_t(maxdims);
}

template <typename T>
inline DataSpace DataSpace::From(const T& value) {
    auto dims = details::inspector<T>::getDimensions(value);
    return DataSpace(dims);
}

template <std::size_t N, std::size_t Width>
inline DataSpace DataSpace::FromCharArrayStrings(const char (&)[N][Width]) {
    return DataSpace(N);
}

namespace details {

/// dimension checks @internal
inline bool checkDimensions(const DataSpace& mem_space, size_t input_dims) {
    size_t dataset_dims = mem_space.getNumberDimensions();
    if (input_dims == dataset_dims)
        return true;

    const std::vector<size_t>& dims = mem_space.getDimensions();
    for (auto i = dims.crbegin(); i != --dims.crend() && *i == 1; ++i)
        --dataset_dims;

    if (input_dims == dataset_dims)
        return true;

    dataset_dims = dims.size();
    for (auto i = dims.cbegin(); i != --dims.cend() && *i == 1; ++i)
        --dataset_dims;

    if (input_dims == dataset_dims)
        return true;

    // The final tests is for scalars
    return input_dims == 0 && dataset_dims == 1 && dims.back() == 1;
}

}  // namespace details
}  // namespace HighFive

#endif  // H5DATASPACE_MISC_HPP

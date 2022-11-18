/*
 *  Copyright (c) 2020 Blue Brain Project
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#pragma once

#include "H5Tpublic.h"
#include "H5Utils.hpp"

namespace HighFive {

namespace details {

template <typename T>
using unqualified_t = typename std::remove_const<typename std::remove_reference<T>::type>::type;

// Find the type of an eventual char array, otherwise void
template <typename>
struct type_char_array {
    using type = void;
};

template <typename T>
struct type_char_array<T*> {
    using type = typename std::conditional<std::is_same<unqualified_t<T>, char>::value,
                                           char*,
                                           typename type_char_array<T>::type>::type;
};

template <typename T, std::size_t N>
struct type_char_array<T[N]> {
    using type = typename std::conditional<std::is_same<unqualified_t<T>, char>::value,
                                           char[N],
                                           typename type_char_array<T>::type>::type;
};

template <typename T>
struct BufferInfo {
    using type_no_const = typename std::remove_const<T>::type;
    using elem_type = typename details::inspector<type_no_const>::base_type;
    using char_array_t = typename details::type_char_array<type_no_const>::type;
    static constexpr bool is_char_array = !std::is_same<char_array_t, void>::value;

    enum Operation { read, write };
    const Operation op;

    template <class F>
    BufferInfo(const DataType& dtype, F getName, Operation _op);

    // member data for info depending on the destination dataset type
    const bool is_fixed_len_string;
    const size_t n_dimensions;
    const DataType data_type;
};

// details implementation
template <typename SrcStrT>
struct string_type_checker {
    static DataType getDataType(const DataType&, const DataType&);
};

template <>
struct string_type_checker<void> {
    inline static DataType getDataType(const DataType& element_type, const DataType& dtype) {
        // TEMP. CHANGE: Ensure that the character set is properly configured to prevent
        // converter issues on HDF5 <=v1.12.0 when loading ASCII strings first.
        // See https://github.com/HDFGroup/hdf5/issues/544 for further information.
        if (H5Tget_class(element_type.getId()) == H5T_STRING &&
            H5Tget_cset(dtype.getId()) == H5T_CSET_ASCII) {
            H5Tset_cset(element_type.getId(), H5T_CSET_ASCII);
        }
        return element_type;
    }
};

template <std::size_t FixedLen>
struct string_type_checker<char[FixedLen]> {
    inline static DataType getDataType(const DataType& element_type, const DataType& dtype) {
        DataType return_type = (dtype.isFixedLenStr()) ? AtomicType<char[FixedLen]>()
                                                       : element_type;
        // TEMP. CHANGE: See string_type_checker<void> definition
        if (H5Tget_cset(dtype.getId()) == H5T_CSET_ASCII) {
            H5Tset_cset(return_type.getId(), H5T_CSET_ASCII);
        }
        return return_type;
    }
};

template <>
struct string_type_checker<char*> {
    inline static DataType getDataType(const DataType&, const DataType& dtype) {
        if (dtype.isFixedLenStr())
            throw DataSetException("Can't output variable-length to fixed-length strings");
        // TEMP. CHANGE: See string_type_checker<void> definition
        DataType return_type = AtomicType<std::string>();
        if (H5Tget_cset(dtype.getId()) == H5T_CSET_ASCII) {
            H5Tset_cset(return_type.getId(), H5T_CSET_ASCII);
        }
        return return_type;
    }
};

template <typename T>
template <class F>
BufferInfo<T>::BufferInfo(const DataType& dtype, F getName, Operation _op)
    : op(_op)
    , is_fixed_len_string(dtype.isFixedLenStr())
    // In case we are using Fixed-len strings we need to subtract one dimension
    , n_dimensions(details::inspector<type_no_const>::recursive_ndim -
                   ((is_fixed_len_string && is_char_array) ? 1 : 0))
    , data_type(
          string_type_checker<char_array_t>::getDataType(create_datatype<elem_type>(), dtype)) {
    if (is_fixed_len_string && std::is_same<elem_type, std::string>::value) {
        throw DataSetException(
            "Can't output std::string as fixed-length. "
            "Use raw arrays or FixedLenStringArray");
    }
    // We warn. In case they are really not convertible an exception will rise on read/write
    if (dtype.getClass() != data_type.getClass()) {
        std::cerr << "HighFive WARNING \"" << getName()
                  << "\": data and hdf5 dataset have different types: " << data_type.string()
                  << " -> " << dtype.string() << std::endl;
    } else if ((dtype.getClass() & data_type.getClass()) == DataTypeClass::Float) {
        if ((op == read) && (dtype.getSize() > data_type.getSize())) {
            std::clog << "HighFive WARNING \"" << getName()
                      << "\": hdf5 dataset has higher floating point precision than data on read: "
                      << dtype.string() << " -> " << data_type.string() << std::endl;
        } else if ((op == write) && (dtype.getSize() < data_type.getSize())) {
            std::clog << "HighFive WARNING \"" << getName()
                      << "\": data has higher floating point precision than hdf5 dataset on write: "
                      << data_type.string() << " -> " << dtype.string() << std::endl;
        }
    }
}

}  // namespace details

}  // namespace HighFive

/*
 *  Copyright (c), 2020, EPFL - Blue Brain Project
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#pragma once

#include <H5Ipublic.h>

#include "H5Utils.hpp"
#include "H5Path_traits.hpp"

namespace HighFive {

template <typename Derivate>
inline PathTraits<Derivate>::PathTraits() {
    static_assert(std::is_same<Derivate, Group>::value || std::is_same<Derivate, DataSet>::value ||
                      std::is_same<Derivate, Attribute>::value,
                  "PathTraits can only be applied to Group, DataSet and Attribute");
    const auto& obj = static_cast<const Derivate&>(*this);
    if (!obj.isValid()) {
        return;
    }
    const hid_t file_id = H5Iget_file_id(obj.getId());
    if (file_id < 0) {
        HDF5ErrMapper::ToException<PropertyException>("getFile(): Could not obtain file of object");
    }
    _file_obj.reset(new File(file_id));
}

template <typename Derivate>
inline std::string PathTraits<Derivate>::getPath() const {
    return details::get_name([this](char* buffer, size_t length) {
        return H5Iget_name(static_cast<const Derivate*>(this)->getId(), buffer, length);
    });
}

template <typename Derivate>
inline File& PathTraits<Derivate>::getFile() const noexcept {
    return *_file_obj;
}

}  // namespace HighFive

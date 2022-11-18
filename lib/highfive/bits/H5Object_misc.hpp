/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5OBJECT_MISC_HPP
#define H5OBJECT_MISC_HPP

#include <iostream>

namespace HighFive {

inline Object::Object()
    : _hid(H5I_INVALID_HID) {}

inline Object::Object(hid_t hid)
    : _hid(hid) {}

inline Object::Object(const Object& other)
    : _hid(other._hid) {
    if (other.isValid() && H5Iinc_ref(_hid) < 0) {
        throw ObjectException("Reference counter increase failure");
    }
}

inline Object::Object(Object&& other) noexcept
    : _hid(other._hid) {
    other._hid = H5I_INVALID_HID;
}

inline Object& Object::operator=(const Object& other) {
    if (this != &other) {
        if (isValid())
            H5Idec_ref(_hid);

        _hid = other._hid;
        if (other.isValid() && H5Iinc_ref(_hid) < 0) {
            throw ObjectException("Reference counter increase failure");
        }
    }
    return *this;
}

inline Object::~Object() {
    if (isValid() && H5Idec_ref(_hid) < 0) {
        std::cerr << "HighFive::~Object: reference counter decrease failure" << std::endl;
    }
}

inline bool Object::isValid() const noexcept {
    return (_hid != H5I_INVALID_HID) && (H5Iis_valid(_hid) != false);
}

inline hid_t Object::getId() const noexcept {
    return _hid;
}

static inline ObjectType _convert_object_type(const H5I_type_t& h5type) {
    switch (h5type) {
    case H5I_FILE:
        return ObjectType::File;
    case H5I_GROUP:
        return ObjectType::Group;
    case H5I_DATATYPE:
        return ObjectType::UserDataType;
    case H5I_DATASPACE:
        return ObjectType::DataSpace;
    case H5I_DATASET:
        return ObjectType::Dataset;
    case H5I_ATTR:
        return ObjectType::Attribute;
    default:
        return ObjectType::Other;
    }
}

inline ObjectType Object::getType() const {
    // H5Iget_type is a very lightweight func which extracts the type from the id
    H5I_type_t h5type;
    if ((h5type = H5Iget_type(_hid)) == H5I_BADID) {
        HDF5ErrMapper::ToException<ObjectException>("Invalid hid or object type");
    }
    return _convert_object_type(h5type);
}

inline ObjectInfo Object::getInfo() const {
    ObjectInfo info;
#if (H5Oget_info_vers < 3)
    if (H5Oget_info(_hid, &info.raw_info) < 0) {
#else
    if (H5Oget_info1(_hid, &info.raw_info) < 0) {
#endif
        HDF5ErrMapper::ToException<ObjectException>("Unable to obtain info for object");
    }
    return info;
}

inline haddr_t ObjectInfo::getAddress() const noexcept {
    return raw_info.addr;
}
inline size_t ObjectInfo::getRefCount() const noexcept {
    return raw_info.rc;
}
inline time_t ObjectInfo::getCreationTime() const noexcept {
    return raw_info.btime;
}
inline time_t ObjectInfo::getModificationTime() const noexcept {
    return raw_info.mtime;
}


}  // namespace HighFive

#endif  // H5OBJECT_MISC_HPP

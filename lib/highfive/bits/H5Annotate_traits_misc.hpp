/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5ANNOTATE_TRAITS_MISC_HPP
#define H5ANNOTATE_TRAITS_MISC_HPP

#include <string>
#include <vector>

#include <H5Apublic.h>
#include <H5Ppublic.h>

#include "H5Attribute_misc.hpp"
#include "H5Iterables_misc.hpp"

namespace HighFive {

template <typename Derivate>
inline Attribute AnnotateTraits<Derivate>::createAttribute(const std::string& attribute_name,
                                                           const DataSpace& space,
                                                           const DataType& dtype) {
    auto attr_id = H5Acreate2(static_cast<Derivate*>(this)->getId(),
                              attribute_name.c_str(),
                              dtype._hid,
                              space._hid,
                              H5P_DEFAULT,
                              H5P_DEFAULT);
    if (attr_id < 0) {
        HDF5ErrMapper::ToException<AttributeException>(
            std::string("Unable to create the attribute \"") + attribute_name + "\":");
    }
    return Attribute(attr_id);
}

template <typename Derivate>
template <typename Type>
inline Attribute AnnotateTraits<Derivate>::createAttribute(const std::string& attribute_name,
                                                           const DataSpace& space) {
    return createAttribute(attribute_name, space, create_and_check_datatype<Type>());
}

template <typename Derivate>
template <typename T>
inline Attribute AnnotateTraits<Derivate>::createAttribute(const std::string& attribute_name,
                                                           const T& data) {
    Attribute att =
        createAttribute(attribute_name,
                        DataSpace::From(data),
                        create_and_check_datatype<typename details::inspector<T>::base_type>());
    att.write(data);
    return att;
}

template <typename Derivate>
inline void AnnotateTraits<Derivate>::deleteAttribute(const std::string& attribute_name) {
    if (H5Adelete(static_cast<const Derivate*>(this)->getId(), attribute_name.c_str()) < 0) {
        HDF5ErrMapper::ToException<AttributeException>(
            std::string("Unable to delete attribute \"") + attribute_name + "\":");
    }
}

template <typename Derivate>
inline Attribute AnnotateTraits<Derivate>::getAttribute(const std::string& attribute_name) const {
    const auto attr_id =
        H5Aopen(static_cast<const Derivate*>(this)->getId(), attribute_name.c_str(), H5P_DEFAULT);
    if (attr_id < 0) {
        HDF5ErrMapper::ToException<AttributeException>(
            std::string("Unable to open the attribute \"") + attribute_name + "\":");
    }
    return Attribute(attr_id);
}

template <typename Derivate>
inline size_t AnnotateTraits<Derivate>::getNumberAttributes() const {
    int res = H5Aget_num_attrs(static_cast<const Derivate*>(this)->getId());
    if (res < 0) {
        HDF5ErrMapper::ToException<AttributeException>(
            std::string("Unable to count attributes in existing group or file"));
    }
    return static_cast<size_t>(res);
}

template <typename Derivate>
inline std::vector<std::string> AnnotateTraits<Derivate>::listAttributeNames() const {
    std::vector<std::string> names;
    details::HighFiveIterateData iterateData(names);

    size_t num_objs = getNumberAttributes();
    names.reserve(num_objs);

    if (H5Aiterate2(static_cast<const Derivate*>(this)->getId(),
                    H5_INDEX_NAME,
                    H5_ITER_INC,
                    NULL,
                    &details::internal_high_five_iterate<H5A_info_t>,
                    static_cast<void*>(&iterateData)) < 0) {
        HDF5ErrMapper::ToException<AttributeException>(
            std::string("Unable to list attributes in group"));
    }

    return names;
}

template <typename Derivate>
inline bool AnnotateTraits<Derivate>::hasAttribute(const std::string& attr_name) const {
    int res = H5Aexists(static_cast<const Derivate*>(this)->getId(), attr_name.c_str());
    if (res < 0) {
        HDF5ErrMapper::ToException<AttributeException>(
            std::string("Unable to check for attribute in group"));
    }
    return res;
}

}  // namespace HighFive

#endif  // H5ANNOTATE_TRAITS_MISC_HPP

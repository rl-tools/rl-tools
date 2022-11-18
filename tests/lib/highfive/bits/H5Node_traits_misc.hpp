/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5NODE_TRAITS_MISC_HPP
#define H5NODE_TRAITS_MISC_HPP

#include <string>
#include <vector>

#include <H5Apublic.h>
#include <H5Dpublic.h>
#include <H5Fpublic.h>
#include <H5Gpublic.h>
#include <H5Ppublic.h>
#include <H5Tpublic.h>

#include "../H5DataSet.hpp"
#include "../H5Group.hpp"
#include "../H5Selection.hpp"
#include "../H5Utility.hpp"
#include "H5DataSet_misc.hpp"
#include "H5Iterables_misc.hpp"
#include "H5Selection_misc.hpp"
#include "H5Slice_traits_misc.hpp"

namespace HighFive {


template <typename Derivate>
inline DataSet NodeTraits<Derivate>::createDataSet(const std::string& dataset_name,
                                                   const DataSpace& space,
                                                   const DataType& dtype,
                                                   const DataSetCreateProps& createProps,
                                                   const DataSetAccessProps& accessProps,
                                                   bool parents) {
    LinkCreateProps lcpl;
    lcpl.add(CreateIntermediateGroup(parents));
    const auto hid = H5Dcreate2(static_cast<Derivate*>(this)->getId(),
                                dataset_name.c_str(),
                                dtype._hid,
                                space._hid,
                                lcpl.getId(),
                                createProps.getId(),
                                accessProps.getId());
    if (hid < 0) {
        HDF5ErrMapper::ToException<DataSetException>(
            std::string("Unable to create the dataset \"") + dataset_name + "\":");
    }
    return DataSet(hid);
}

template <typename Derivate>
template <typename Type>
inline DataSet NodeTraits<Derivate>::createDataSet(const std::string& dataset_name,
                                                   const DataSpace& space,
                                                   const DataSetCreateProps& createProps,
                                                   const DataSetAccessProps& accessProps,
                                                   bool parents) {
    return createDataSet(
        dataset_name, space, create_and_check_datatype<Type>(), createProps, accessProps, parents);
}

template <typename Derivate>
template <typename T>
inline DataSet NodeTraits<Derivate>::createDataSet(const std::string& dataset_name,
                                                   const T& data,
                                                   const DataSetCreateProps& createProps,
                                                   const DataSetAccessProps& accessProps,
                                                   bool parents) {
    DataSet ds =
        createDataSet(dataset_name,
                      DataSpace::From(data),
                      create_and_check_datatype<typename details::inspector<T>::base_type>(),
                      createProps,
                      accessProps,
                      parents);
    ds.write(data);
    return ds;
}

template <typename Derivate>
template <std::size_t N>
inline DataSet NodeTraits<Derivate>::createDataSet(const std::string& dataset_name,
                                                   const FixedLenStringArray<N>& data,
                                                   const DataSetCreateProps& createProps,
                                                   const DataSetAccessProps& accessProps,
                                                   bool parents) {
    DataSet ds = createDataSet<char[N]>(
        dataset_name, DataSpace(data.size()), createProps, accessProps, parents);
    ds.write(data);
    return ds;
}

template <typename Derivate>
inline DataSet NodeTraits<Derivate>::getDataSet(const std::string& dataset_name,
                                                const DataSetAccessProps& accessProps) const {
    const auto hid = H5Dopen2(static_cast<const Derivate*>(this)->getId(),
                              dataset_name.c_str(),
                              accessProps.getId());
    if (hid < 0) {
        HDF5ErrMapper::ToException<DataSetException>(std::string("Unable to open the dataset \"") +
                                                     dataset_name + "\":");
    }
    return DataSet(hid);
}

template <typename Derivate>
inline Group NodeTraits<Derivate>::createGroup(const std::string& group_name, bool parents) {
    LinkCreateProps lcpl;
    lcpl.add(CreateIntermediateGroup(parents));
    const auto hid = H5Gcreate2(static_cast<Derivate*>(this)->getId(),
                                group_name.c_str(),
                                lcpl.getId(),
                                H5P_DEFAULT,
                                H5P_DEFAULT);
    if (hid < 0) {
        HDF5ErrMapper::ToException<GroupException>(std::string("Unable to create the group \"") +
                                                   group_name + "\":");
    }
    return Group(hid);
}

template <typename Derivate>
inline Group NodeTraits<Derivate>::createGroup(const std::string& group_name,
                                               const GroupCreateProps& createProps,
                                               bool parents) {
    LinkCreateProps lcpl;
    lcpl.add(CreateIntermediateGroup(parents));
    const auto hid = H5Gcreate2(static_cast<Derivate*>(this)->getId(),
                                group_name.c_str(),
                                lcpl.getId(),
                                createProps.getId(),
                                H5P_DEFAULT);
    if (hid < 0) {
        HDF5ErrMapper::ToException<GroupException>(std::string("Unable to create the group \"") +
                                                   group_name + "\":");
    }
    return Group(hid);
}

template <typename Derivate>
inline Group NodeTraits<Derivate>::getGroup(const std::string& group_name) const {
    const auto hid =
        H5Gopen2(static_cast<const Derivate*>(this)->getId(), group_name.c_str(), H5P_DEFAULT);
    if (hid < 0) {
        HDF5ErrMapper::ToException<GroupException>(std::string("Unable to open the group \"") +
                                                   group_name + "\":");
    }
    return Group(hid);
}

template <typename Derivate>
inline size_t NodeTraits<Derivate>::getNumberObjects() const {
    hsize_t res;
    if (H5Gget_num_objs(static_cast<const Derivate*>(this)->getId(), &res) < 0) {
        HDF5ErrMapper::ToException<GroupException>(
            std::string("Unable to count objects in existing group or file"));
    }
    return static_cast<size_t>(res);
}

template <typename Derivate>
inline std::string NodeTraits<Derivate>::getObjectName(size_t index) const {
    return details::get_name([&](char* buffer, size_t length) {
        return H5Lget_name_by_idx(static_cast<const Derivate*>(this)->getId(),
                                  ".",
                                  H5_INDEX_NAME,
                                  H5_ITER_INC,
                                  index,
                                  buffer,
                                  length,
                                  H5P_DEFAULT);
    });
}

template <typename Derivate>
inline bool NodeTraits<Derivate>::rename(const std::string& src_path,
                                         const std::string& dst_path,
                                         bool parents) const {
    LinkCreateProps lcpl;
    lcpl.add(CreateIntermediateGroup(parents));
    herr_t status = H5Lmove(static_cast<const Derivate*>(this)->getId(),
                            src_path.c_str(),
                            static_cast<const Derivate*>(this)->getId(),
                            dst_path.c_str(),
                            lcpl.getId(),
                            H5P_DEFAULT);
    if (status < 0) {
        HDF5ErrMapper::ToException<GroupException>(std::string("Unable to move link to \"") +
                                                   dst_path + "\":");
        return false;
    }
    return true;
}

template <typename Derivate>
inline std::vector<std::string> NodeTraits<Derivate>::listObjectNames() const {
    std::vector<std::string> names;
    details::HighFiveIterateData iterateData(names);

    size_t num_objs = getNumberObjects();
    names.reserve(num_objs);

    if (H5Literate(static_cast<const Derivate*>(this)->getId(),
                   H5_INDEX_NAME,
                   H5_ITER_INC,
                   NULL,
                   &details::internal_high_five_iterate<H5L_info_t>,
                   static_cast<void*>(&iterateData)) < 0) {
        HDF5ErrMapper::ToException<GroupException>(std::string("Unable to list objects in group"));
    }

    return names;
}

template <typename Derivate>
inline bool NodeTraits<Derivate>::_exist(const std::string& node_name, bool raise_errors) const {
    SilenceHDF5 silencer{};
    const auto val =
        H5Lexists(static_cast<const Derivate*>(this)->getId(), node_name.c_str(), H5P_DEFAULT);
    if (val < 0) {
        if (raise_errors) {
            HDF5ErrMapper::ToException<GroupException>("Invalid link for exist()");
        } else {
            return false;
        }
    }

    // The root path always exists, but H5Lexists return 0 or 1
    // depending of the version of HDF5, so always return true for it
    // We had to call H5Lexists anyway to check that there are no errors
    return (node_name == "/") ? true : (val > 0);
}

template <typename Derivate>
inline bool NodeTraits<Derivate>::exist(const std::string& group_path) const {
    // When there are slashes, first check everything is fine
    // so that subsequent errors are only due to missing intermediate groups
    if (group_path.find('/') != std::string::npos) {
        _exist("/");  // Shall not throw under normal circumstances
        // Unless "/" (already checked), verify path exists (not throwing errors)
        return (group_path == "/") ? true : _exist(group_path, false);
    }
    return _exist(group_path);
}


template <typename Derivate>
inline void NodeTraits<Derivate>::unlink(const std::string& node_name) const {
    const herr_t val =
        H5Ldelete(static_cast<const Derivate*>(this)->getId(), node_name.c_str(), H5P_DEFAULT);
    if (val < 0) {
        HDF5ErrMapper::ToException<GroupException>(std::string("Invalid name for unlink() "));
    }
}


// convert internal link types to enum class.
// This function is internal, so H5L_TYPE_ERROR shall be handled in the calling context
static inline LinkType _convert_link_type(const H5L_type_t& ltype) noexcept {
    switch (ltype) {
    case H5L_TYPE_HARD:
        return LinkType::Hard;
    case H5L_TYPE_SOFT:
        return LinkType::Soft;
    case H5L_TYPE_EXTERNAL:
        return LinkType::External;
    default:
        // Other link types are possible but are considered strange to HighFive.
        // see https://support.hdfgroup.org/HDF5/doc/RM/H5L/H5Lregister.htm
        return LinkType::Other;
    }
}

template <typename Derivate>
inline LinkType NodeTraits<Derivate>::getLinkType(const std::string& node_name) const {
    H5L_info_t linkinfo;
    if (H5Lget_info(static_cast<const Derivate*>(this)->getId(),
                    node_name.c_str(),
                    &linkinfo,
                    H5P_DEFAULT) < 0 ||
        linkinfo.type == H5L_TYPE_ERROR) {
        HDF5ErrMapper::ToException<GroupException>(std::string("Unable to obtain info for link ") +
                                                   node_name);
    }
    return _convert_link_type(linkinfo.type);
}

template <typename Derivate>
inline ObjectType NodeTraits<Derivate>::getObjectType(const std::string& node_name) const {
    return _open(node_name).getType();
}


template <typename Derivate>
inline void NodeTraits<Derivate>::createSoftLink(const std::string& link_name,
                                                 const std::string& obj_path,
                                                 LinkCreateProps linkCreateProps,
                                                 const LinkAccessProps& linkAccessProps,
                                                 const bool parents) {
    if (parents) {
        linkCreateProps.add(CreateIntermediateGroup{});
    }
    auto status = H5Lcreate_soft(obj_path.c_str(),
                                 static_cast<const Derivate*>(this)->getId(),
                                 link_name.c_str(),
                                 linkCreateProps.getId(),
                                 linkAccessProps.getId());
    if (status < 0) {
        HDF5ErrMapper::ToException<GroupException>(std::string("Unable to create soft link: "));
    }
}


template <typename Derivate>
inline void NodeTraits<Derivate>::createExternalLink(const std::string& link_name,
                                                     const std::string& h5_file,
                                                     const std::string& obj_path,
                                                     LinkCreateProps linkCreateProps,
                                                     const LinkAccessProps& linkAccessProps,
                                                     const bool parents) {
    if (parents) {
        linkCreateProps.add(CreateIntermediateGroup{});
    }
    auto status = H5Lcreate_external(h5_file.c_str(),
                                     obj_path.c_str(),
                                     static_cast<const Derivate*>(this)->getId(),
                                     link_name.c_str(),
                                     linkCreateProps.getId(),
                                     linkAccessProps.getId());
    if (status < 0) {
        HDF5ErrMapper::ToException<GroupException>(std::string("Unable to create external link: "));
    }
}


template <typename Derivate>
inline Object NodeTraits<Derivate>::_open(const std::string& node_name,
                                          const DataSetAccessProps& accessProps) const {
    const auto id = H5Oopen(static_cast<const Derivate*>(this)->getId(),
                            node_name.c_str(),
                            accessProps.getId());
    if (id < 0) {
        HDF5ErrMapper::ToException<GroupException>(std::string("Unable to open \"") + node_name +
                                                   "\":");
    }
    return Object(id);
}


}  // namespace HighFive

#endif  // H5NODE_TRAITS_MISC_HPP

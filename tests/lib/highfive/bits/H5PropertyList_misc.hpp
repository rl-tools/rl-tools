/*
 *  Copyright (c), 2017-2018, Adrien Devresse <adrien.devresse@epfl.ch>
 *                            Juan Hernando <juan.hernando@epfl.ch>
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5PROPERTY_LIST_MISC_HPP
#define H5PROPERTY_LIST_MISC_HPP

#include <H5Ppublic.h>

namespace HighFive {

namespace {
inline hid_t convert_plist_type(PropertyType propertyType) {
    // The HP5_XXX are macros with function calls so we can't assign
    // them as the enum values
    switch (propertyType) {
    case PropertyType::OBJECT_CREATE:
        return H5P_OBJECT_CREATE;
    case PropertyType::FILE_CREATE:
        return H5P_FILE_CREATE;
    case PropertyType::FILE_ACCESS:
        return H5P_FILE_ACCESS;
    case PropertyType::DATASET_CREATE:
        return H5P_DATASET_CREATE;
    case PropertyType::DATASET_ACCESS:
        return H5P_DATASET_ACCESS;
    case PropertyType::DATASET_XFER:
        return H5P_DATASET_XFER;
    case PropertyType::GROUP_CREATE:
        return H5P_GROUP_CREATE;
    case PropertyType::GROUP_ACCESS:
        return H5P_GROUP_ACCESS;
    case PropertyType::DATATYPE_CREATE:
        return H5P_DATATYPE_CREATE;
    case PropertyType::DATATYPE_ACCESS:
        return H5P_DATATYPE_ACCESS;
    case PropertyType::STRING_CREATE:
        return H5P_STRING_CREATE;
    case PropertyType::ATTRIBUTE_CREATE:
        return H5P_ATTRIBUTE_CREATE;
    case PropertyType::OBJECT_COPY:
        return H5P_OBJECT_COPY;
    case PropertyType::LINK_CREATE:
        return H5P_LINK_CREATE;
    case PropertyType::LINK_ACCESS:
        return H5P_LINK_ACCESS;
    default:
        HDF5ErrMapper::ToException<PropertyException>("Unsupported property list type");
    }
}

}  // namespace


inline PropertyListBase::PropertyListBase() noexcept
    : Object(H5P_DEFAULT) {}


template <PropertyType T>
inline void PropertyList<T>::_initializeIfNeeded() {
    if (_hid != H5P_DEFAULT) {
        return;
    }
    if ((_hid = H5Pcreate(convert_plist_type(T))) < 0) {
        HDF5ErrMapper::ToException<PropertyException>("Unable to create property list");
    }
}

template <PropertyType T>
template <typename P>
inline void PropertyList<T>::add(const P& property) {
    _initializeIfNeeded();
    property.apply(_hid);
}

template <PropertyType T>
template <typename F, typename... Args>
inline void RawPropertyList<T>::add(const F& funct, const Args&... args) {
    this->_initializeIfNeeded();
    if (funct(this->_hid, args...) < 0) {
        HDF5ErrMapper::ToException<PropertyException>("Error setting raw hdf5 property.");
    }
}


// Specific options to be added to Property Lists
#if H5_VERSION_GE(1, 10, 1)
inline FileSpaceStrategy::FileSpaceStrategy(H5F_fspace_strategy_t strategy,
                                            hbool_t persist,
                                            hsize_t threshold)
    : _strategy(strategy)
    , _persist(persist)
    , _threshold(threshold) {}

inline void FileSpaceStrategy::apply(const hid_t list) const {
    if (H5Pset_file_space_strategy(list, _strategy, _persist, _threshold) < 0) {
        HDF5ErrMapper::ToException<PropertyException>("Error setting file space strategy.");
    }
}

inline FileSpacePageSize::FileSpacePageSize(hsize_t page_size)
    : _page_size(page_size) {}

inline void FileSpacePageSize::apply(const hid_t list) const {
    if (H5Pset_file_space_page_size(list, _page_size) < 0) {
        HDF5ErrMapper::ToException<PropertyException>("Error setting file space page size.");
    }
}

#ifndef H5_HAVE_PARALLEL
inline PageBufferSize::PageBufferSize(size_t page_buffer_size,
                                      unsigned min_meta_percent,
                                      unsigned min_raw_percent)
    : _page_buffer_size(page_buffer_size)
    , _min_meta(min_meta_percent)
    , _min_raw(min_raw_percent) {}

inline void PageBufferSize::apply(const hid_t list) const {
    if (H5Pset_page_buffer_size(list, _page_buffer_size, _min_meta, _min_raw) < 0) {
        HDF5ErrMapper::ToException<PropertyException>("Error setting page buffer size.");
    }
}
#endif
#endif

#ifdef H5_HAVE_PARALLEL

inline void MPIOCollectiveMetadata::apply(const hid_t plist) const {
    auto read = MPIOCollectiveMetadataRead{collective_};
    auto write = MPIOCollectiveMetadataWrite{collective_};

    read.apply(plist);
    write.apply(plist);
}

inline void MPIOCollectiveMetadataRead::apply(const hid_t plist) const {
    if (H5Pset_all_coll_metadata_ops(plist, collective_) < 0) {
        HDF5ErrMapper::ToException<FileException>("Unable to request collective metadata reads");
    }
}

inline void MPIOCollectiveMetadataWrite::apply(const hid_t plist) const {
    if (H5Pset_coll_metadata_write(plist, collective_) < 0) {
        HDF5ErrMapper::ToException<FileException>("Unable to request collective metadata writes");
    }
}

#endif


inline void EstimatedLinkInfo::apply(const hid_t hid) const {
    if (H5Pset_est_link_info(hid, _entries, _length) < 0) {
        HDF5ErrMapper::ToException<PropertyException>("Error setting estimated link info");
    }
}

inline void Chunking::apply(const hid_t hid) const {
    if (H5Pset_chunk(hid, static_cast<int>(_dims.size()), _dims.data()) < 0) {
        HDF5ErrMapper::ToException<PropertyException>("Error setting chunk property");
    }
}

inline void Deflate::apply(const hid_t hid) const {
    if (!H5Zfilter_avail(H5Z_FILTER_DEFLATE) || H5Pset_deflate(hid, _level) < 0) {
        HDF5ErrMapper::ToException<PropertyException>("Error setting deflate property");
    }
}

inline void Szip::apply(const hid_t hid) const {
    if (!H5Zfilter_avail(H5Z_FILTER_SZIP)) {
        HDF5ErrMapper::ToException<PropertyException>("Error setting szip property");
    }

    if (H5Pset_szip(hid, _options_mask, _pixels_per_block) < 0) {
        HDF5ErrMapper::ToException<PropertyException>("Error setting szip property");
    }
}

inline void Shuffle::apply(const hid_t hid) const {
    if (!H5Zfilter_avail(H5Z_FILTER_SHUFFLE)) {
        HDF5ErrMapper::ToException<PropertyException>("Error setting shuffle property");
    }

    if (H5Pset_shuffle(hid) < 0) {
        HDF5ErrMapper::ToException<PropertyException>("Error setting shuffle property");
    }
}

inline void AllocationTime::apply(hid_t dcpl) const {
    if (H5Pset_alloc_time(dcpl, _alloc_time) < 0) {
        HDF5ErrMapper::ToException<PropertyException>("Error setting allocation time");
    }
}

inline void Caching::apply(const hid_t hid) const {
    if (H5Pset_chunk_cache(hid, _numSlots, _cacheSize, _w0) < 0) {
        HDF5ErrMapper::ToException<PropertyException>("Error setting dataset cache parameters");
    }
}

inline void CreateIntermediateGroup::apply(const hid_t hid) const {
    if (H5Pset_create_intermediate_group(hid, _create ? 1 : 0) < 0) {
        HDF5ErrMapper::ToException<PropertyException>(
            "Error setting property for create intermediate groups");
    }
}

#ifdef H5_HAVE_PARALLEL
inline void UseCollectiveIO::apply(const hid_t hid) const {
    if (H5Pset_dxpl_mpio(hid, _enable ? H5FD_MPIO_COLLECTIVE : H5FD_MPIO_INDEPENDENT) < 0) {
        HDF5ErrMapper::ToException<PropertyException>("Error setting H5Pset_dxpl_mpio.");
    }
}
#endif

}  // namespace HighFive

#endif  // H5PROPERTY_LIST_HPP

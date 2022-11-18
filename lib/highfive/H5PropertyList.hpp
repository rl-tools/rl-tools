/*
 *  Copyright (c), 2017-2018, Adrien Devresse <adrien.devresse@epfl.ch>
 *                            Juan Hernando <juan.hernando@epfl.ch>
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5PROPERTY_LIST_HPP
#define H5PROPERTY_LIST_HPP

#include <vector>

#include <H5Ppublic.h>

// Required by MPIOFileAccess
#ifdef H5_HAVE_PARALLEL
#include <H5FDmpi.h>
#endif

#include "H5Exception.hpp"
#include "H5Object.hpp"

namespace HighFive {

///
/// \brief Types of property lists
///
enum class PropertyType : int {
    OBJECT_CREATE,
    FILE_CREATE,
    FILE_ACCESS,
    DATASET_CREATE,
    DATASET_ACCESS,
    DATASET_XFER,
    GROUP_CREATE,
    GROUP_ACCESS,
    DATATYPE_CREATE,
    DATATYPE_ACCESS,
    STRING_CREATE,
    ATTRIBUTE_CREATE,
    OBJECT_COPY,
    LINK_CREATE,
    LINK_ACCESS,
};

namespace details {
template <typename T, typename U>
T get_plist(const U& obj, hid_t (*f)(hid_t)) {
    auto hid = f(obj.getId());
    if (hid < 0) {
        HDF5ErrMapper::ToException<PropertyException>(std::string("Unable to get property list"));
    }
    T t{};
    t._hid = hid;
    return t;
}
}  // namespace details

///
/// \brief Base Class for Property lists, providing global default
class PropertyListBase: public Object {
  public:
    PropertyListBase() noexcept;

    static const PropertyListBase& Default() noexcept {
        static const PropertyListBase plist{};
        return plist;
    }

  private:
    template <typename T, typename U>
    friend T details::get_plist(const U&, hid_t (*f)(hid_t));
};

///
/// \brief HDF5 property Lists
///
template <PropertyType T>
class PropertyList: public PropertyListBase {
  public:
    ///
    /// \brief return the type of this PropertyList
    constexpr PropertyType getType() const noexcept {
        return T;
    }

    ///
    /// Add a property to this property list.
    /// A property is an object which is expected to have a method with the
    /// following signature void apply(hid_t hid) const
    ///
    template <typename P>
    void add(const P& property);

    ///
    /// Return the Default property type object
    static const PropertyList<T>& Default() noexcept {
        return static_cast<const PropertyList<T>&>(PropertyListBase::Default());
    }

  protected:
    void _initializeIfNeeded();
};

using ObjectCreateProps = PropertyList<PropertyType::OBJECT_CREATE>;
using FileCreateProps = PropertyList<PropertyType::FILE_CREATE>;
using FileAccessProps = PropertyList<PropertyType::FILE_ACCESS>;
using DataSetCreateProps = PropertyList<PropertyType::DATASET_CREATE>;
using DataSetAccessProps = PropertyList<PropertyType::DATASET_ACCESS>;
using DataTransferProps = PropertyList<PropertyType::DATASET_XFER>;
using GroupCreateProps = PropertyList<PropertyType::GROUP_CREATE>;
using GroupAccessProps = PropertyList<PropertyType::GROUP_ACCESS>;
using DataTypeCreateProps = PropertyList<PropertyType::DATATYPE_CREATE>;
using DataTypeAccessProps = PropertyList<PropertyType::DATATYPE_ACCESS>;
using StringCreateProps = PropertyList<PropertyType::STRING_CREATE>;
using AttributeCreateProps = PropertyList<PropertyType::ATTRIBUTE_CREATE>;
using ObjectCopyProps = PropertyList<PropertyType::OBJECT_COPY>;
using LinkCreateProps = PropertyList<PropertyType::LINK_CREATE>;
using LinkAccessProps = PropertyList<PropertyType::LINK_ACCESS>;

///
/// RawPropertyLists are to be used when advanced H5 properties
/// are desired and are not part of the HighFive API.
/// Therefore this class is mainly for internal use.
template <PropertyType T>
class RawPropertyList: public PropertyList<T> {
  public:
    template <typename F, typename... Args>
    void add(const F& funct, const Args&... args);
};

#ifdef H5_HAVE_PARALLEL
///
/// \brief Configure MPI access for the file
///
/// All further modifications to the structure of the file will have to be
/// done with collective operations
///
class MPIOFileAccess {
  public:
    MPIOFileAccess(MPI_Comm comm, MPI_Info info)
        : _comm(comm)
        , _info(info) {}

    void apply(const hid_t list) const {
        if (H5Pset_fapl_mpio(list, _comm, _info) < 0) {
            HDF5ErrMapper::ToException<FileException>("Unable to set-up MPIO Driver configuration");
        }
    }

  private:
    MPI_Comm _comm;
    MPI_Info _info;
};

///
/// \brief Use collective MPI-IO for metadata read and write?
///
/// See `MPIOCollectiveMetadataRead` and `MPIOCollectiveMetadataWrite`.
///
class MPIOCollectiveMetadata {
  public:
    explicit MPIOCollectiveMetadata(bool collective = true)
        : collective_(collective) {}

  private:
    friend FileAccessProps;
    void apply(hid_t plist) const;
    bool collective_;
};

///
/// \brief Use collective MPI-IO for metadata read?
///
/// Note that when used in a file access property list, this will force all reads
/// of meta data to be collective. HDF5 function may implicitly perform metadata
/// reads. These functions would become collective. A list of functions that
/// perform metadata reads can be found in the HDF5 documentation, e.g.
///    https://docs.hdfgroup.org/hdf5/v1_12/group___g_a_c_p_l.html
///
/// In HighFive setting collective read is (currently) only supported on file level.
///
/// Please also consult upstream documentation of `H5Pset_all_coll_metadata_ops`.
///
class MPIOCollectiveMetadataRead {
  public:
    explicit MPIOCollectiveMetadataRead(bool collective = true)
        : collective_(collective) {}

  private:
    friend FileAccessProps;
    friend MPIOCollectiveMetadata;

    void apply(hid_t plist) const;

    bool collective_;
};

///
/// \brief Use collective MPI-IO for metadata write?
///
/// In order to keep the in-memory representation of the file structure
/// consistent across MPI ranks, writing meta data is always a collective
/// operation. Meaning all MPI ranks must participate. Passing this setting
/// enables using MPI-IO collective operations for metadata writes.
///
/// Please also consult upstream documentation of `H5Pset_coll_metadata_write`.
///
class MPIOCollectiveMetadataWrite {
  public:
    explicit MPIOCollectiveMetadataWrite(bool collective = true)
        : collective_(collective) {}

  private:
    friend FileAccessProps;
    friend MPIOCollectiveMetadata;

    void apply(hid_t plist) const;

    bool collective_;
};

#endif

///
/// \brief Configure the version bounds for the file
///
/// Used to define the compatibility of objects created within HDF5 files,
/// and affects the format of groups stored in the file.
///
/// See also the documentation of \c H5P_SET_LIBVER_BOUNDS in HDF5.
///
/// Possible values for \c low and \c high are:
/// * \c H5F_LIBVER_EARLIEST
/// * \c H5F_LIBVER_V18
/// * \c H5F_LIBVER_V110
/// * \c H5F_LIBVER_NBOUNDS
/// * \c H5F_LIBVER_LATEST currently defined as \c H5F_LIBVER_V110 within
///   HDF5
///
class FileVersionBounds {
  public:
    FileVersionBounds(H5F_libver_t low, H5F_libver_t high)
        : _low(low)
        , _high(high) {}

  private:
    friend FileAccessProps;
    void apply(const hid_t list) const {
        if (H5Pset_libver_bounds(list, _low, _high) < 0) {
            HDF5ErrMapper::ToException<PropertyException>("Error setting file version bounds");
        }
    }
    const H5F_libver_t _low;
    const H5F_libver_t _high;
};

///
/// \brief Configure the metadata block size to use writing to files
///
/// \param size Metadata block size in bytes
///
class MetadataBlockSize {
  public:
    MetadataBlockSize(hsize_t size)
        : _size(size) {}

  private:
    friend FileAccessProps;
    void apply(const hid_t list) const {
        if (H5Pset_meta_block_size(list, _size) < 0) {
            HDF5ErrMapper::ToException<PropertyException>("Error setting metadata block size");
        }
    }
    const hsize_t _size;
};

#if H5_VERSION_GE(1, 10, 1)
///
/// \brief Configure the file space strategy.
///
/// See the upstream documentation of `H5Pget_file_space_strategy` for more details. Essentially,
/// it enables configuring how space is allocate in the file.
///
class FileSpaceStrategy {
  public:
    ///
    /// \brief Create a file space strategy property.
    ///
    /// \param strategy The HDF5 free space strategy.
    /// \param persist Should free space managers be persisted across file closing and reopening.
    /// \param threshold The free-space manager wont track sections small than this threshold.
    FileSpaceStrategy(H5F_fspace_strategy_t strategy, hbool_t persist, hsize_t threshold);

  private:
    friend FileCreateProps;

    void apply(const hid_t list) const;

    H5F_fspace_strategy_t _strategy;
    hbool_t _persist;
    hsize_t _threshold;
};

///
/// \brief Configure the page size for paged allocation.
///
/// See the upstream documentation of `H5Pset_file_space_page_size` for more details. Essentially,
/// it enables configuring the page size when paged allocation is used.
///
/// General information about paged allocation can be found in the upstream documentation "RFC: Page
/// Buffering".
///
class FileSpacePageSize {
  public:
    ///
    /// \brief Create a file space strategy property.
    ///
    /// \param page_size The page size in bytes.
    explicit FileSpacePageSize(hsize_t page_size);

  private:
    friend FileCreateProps;

    void apply(const hid_t list) const;

    hsize_t _page_size;
};

#ifndef H5_HAVE_PARALLEL
/// \brief Set size of the page buffer.
///
/// Please, consult the upstream documentation of
///    H5Pset_page_buffer_size
///    H5Pget_page_buffer_size
/// Note that this setting is only valid for page allocated/aggregated
/// files, i.e. those that have file space strategy "Page".
///
/// Tests suggest this doesn't work in the parallel version of the
/// library. Hence, this isn't available at compile time if the parallel
/// library was selected.
class PageBufferSize {
  public:
    /// Property to set page buffer sizes.
    ///
    /// @param page_buffer_size maximum size of the page buffer in bytes.
    /// @param min_meta_percent fraction of the page buffer dedicated to meta data, in percent.
    /// @param min_raw_percent fraction of the page buffer dedicated to raw data, in percent.
    explicit PageBufferSize(size_t page_buffer_size,
                            unsigned min_meta_percent = 0,
                            unsigned min_raw_percent = 0);

  private:
    friend FileAccessProps;

    void apply(hid_t list) const;

    hsize_t _page_buffer_size;
    unsigned _min_meta;
    unsigned _min_raw;
};
#endif
#endif

/// \brief Set hints as to how many links to expect and their average length
///
class EstimatedLinkInfo {
  public:
    explicit EstimatedLinkInfo(unsigned entries, unsigned length)
        : _entries(entries)
        , _length(length) {}

  private:
    friend GroupCreateProps;
    void apply(hid_t hid) const;
    const unsigned _entries;
    const unsigned _length;
};


class Chunking {
  public:
    explicit Chunking(const std::vector<hsize_t>& dims)
        : _dims(dims) {}

    Chunking(const std::initializer_list<hsize_t>& items)
        : Chunking(std::vector<hsize_t>{items}) {}

    template <typename... Args>
    explicit Chunking(hsize_t item, Args... args)
        : Chunking(std::vector<hsize_t>{item, static_cast<hsize_t>(args)...}) {}

    const std::vector<hsize_t>& getDimensions() const noexcept {
        return _dims;
    }

  private:
    friend DataSetCreateProps;
    void apply(hid_t hid) const;
    const std::vector<hsize_t> _dims;
};

class Deflate {
  public:
    explicit Deflate(unsigned level)
        : _level(level) {}

  private:
    friend DataSetCreateProps;
    friend GroupCreateProps;
    void apply(hid_t hid) const;
    const unsigned _level;
};

class Szip {
  public:
    explicit Szip(unsigned options_mask = H5_SZIP_EC_OPTION_MASK,
                  unsigned pixels_per_block = H5_SZIP_MAX_PIXELS_PER_BLOCK)
        : _options_mask(options_mask)
        , _pixels_per_block(pixels_per_block) {}

  private:
    friend DataSetCreateProps;
    void apply(hid_t hid) const;
    const unsigned _options_mask;
    const unsigned _pixels_per_block;
};

class Shuffle {
  public:
    Shuffle() = default;

  private:
    friend DataSetCreateProps;
    void apply(hid_t hid) const;
};

/// \brief When are datasets allocated?
///
/// The precise time of when HDF5 requests space to store the dataset
/// can be configured. Please, consider the upstream documentation for
/// `H5Pset_alloc_time`.
class AllocationTime {
  public:
    explicit AllocationTime(H5D_alloc_time_t alloc_time)
        : _alloc_time(alloc_time) {}

  private:
    friend DataSetCreateProps;
    void apply(hid_t dcpl) const;

    H5D_alloc_time_t _alloc_time;
};

/// Dataset access property to control chunk cache configuration.
/// Do not confuse with the similar file access property for H5Pset_cache
class Caching {
  public:
    /// https://support.hdfgroup.org/HDF5/doc/RM/H5P/H5Pset_chunk_cache.html for
    /// details.
    Caching(const size_t numSlots,
            const size_t cacheSize,
            const double w0 = static_cast<double>(H5D_CHUNK_CACHE_W0_DEFAULT))
        : _numSlots(numSlots)
        , _cacheSize(cacheSize)
        , _w0(w0) {}

  private:
    friend DataSetAccessProps;
    void apply(hid_t hid) const;
    const size_t _numSlots;
    const size_t _cacheSize;
    const double _w0;
};

class CreateIntermediateGroup {
  public:
    explicit CreateIntermediateGroup(bool create = true)
        : _create(create) {}

  private:
    friend ObjectCreateProps;
    friend LinkCreateProps;
    void apply(hid_t hid) const;
    const bool _create;
};

#ifdef H5_HAVE_PARALLEL
class UseCollectiveIO {
  public:
    explicit UseCollectiveIO(bool enable = true)
        : _enable(enable) {}

  private:
    friend DataTransferProps;
    void apply(hid_t hid) const;
    bool _enable;
};
#endif

}  // namespace HighFive

#include "bits/H5PropertyList_misc.hpp"

#endif  // H5PROPERTY_LIST_HPP

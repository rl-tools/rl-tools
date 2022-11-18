/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5FILE_MISC_HPP
#define H5FILE_MISC_HPP

#include <string>

#include <H5Fpublic.h>

#include "../H5Utility.hpp"
#include "H5Utils.hpp"

namespace HighFive {

namespace {  // unnamed

// libhdf5 uses a preprocessor trick on their oflags
// we can not declare them constant without a mapper
inline unsigned convert_open_flag(unsigned openFlags) {
    unsigned res_open = 0;
    if (openFlags & File::ReadOnly)
        res_open |= H5F_ACC_RDONLY;
    if (openFlags & File::ReadWrite)
        res_open |= H5F_ACC_RDWR;
    if (openFlags & File::Create)
        res_open |= H5F_ACC_CREAT;
    if (openFlags & File::Truncate)
        res_open |= H5F_ACC_TRUNC;
    if (openFlags & File::Excl)
        res_open |= H5F_ACC_EXCL;
    return res_open;
}
}  // namespace

inline File::File(const std::string& filename,
                  unsigned openFlags,
                  const FileAccessProps& fileAccessProps)
    : File(filename, openFlags, FileCreateProps::Default(), fileAccessProps) {}


inline File::File(const std::string& filename,
                  unsigned openFlags,
                  const FileCreateProps& fileCreateProps,
                  const FileAccessProps& fileAccessProps) {
    openFlags = convert_open_flag(openFlags);

    unsigned createMode = openFlags & (H5F_ACC_TRUNC | H5F_ACC_EXCL);
    unsigned openMode = openFlags & (H5F_ACC_RDWR | H5F_ACC_RDONLY);
    bool mustCreate = createMode > 0;
    bool openOrCreate = (openFlags & H5F_ACC_CREAT) > 0;

    // open is default. It's skipped only if flags require creation
    // If open fails it will try create() if H5F_ACC_CREAT is set
    if (!mustCreate) {
        // Silence open errors if create is allowed
        std::unique_ptr<SilenceHDF5> silencer;
        if (openOrCreate)
            silencer.reset(new SilenceHDF5());

        _hid = H5Fopen(filename.c_str(), openMode, fileAccessProps.getId());

        if (isValid())
            return;  // Done

        if (openOrCreate) {
            // Will attempt to create ensuring wont clobber any file
            createMode = H5F_ACC_EXCL;
        } else {
            HDF5ErrMapper::ToException<FileException>(
                std::string("Unable to open file " + filename));
        }
    }

    auto fcpl = fileCreateProps.getId();
    auto fapl = fileAccessProps.getId();
    if ((_hid = H5Fcreate(filename.c_str(), createMode, fcpl, fapl)) < 0) {
        HDF5ErrMapper::ToException<FileException>(std::string("Unable to create file " + filename));
    }
}

inline const std::string& File::getName() const noexcept {
    if (_filename.empty()) {
        _filename = details::get_name(
            [this](char* buffer, size_t length) { return H5Fget_name(getId(), buffer, length); });
    }
    return _filename;
}

inline hsize_t File::getMetadataBlockSize() const {
    hsize_t size;
    auto fapl = getAccessPropertyList();
    if (H5Pget_meta_block_size(fapl.getId(), &size) < 0) {
        HDF5ErrMapper::ToException<FileException>(
            std::string("Unable to access file metadata block size"));
    }
    return size;
}

inline std::pair<H5F_libver_t, H5F_libver_t> File::getVersionBounds() const {
    H5F_libver_t low;
    H5F_libver_t high;
    auto fapl = getAccessPropertyList();
    if (H5Pget_libver_bounds(fapl.getId(), &low, &high) < 0) {
        HDF5ErrMapper::ToException<FileException>(
            std::string("Unable to access file version bounds"));
    }
    return std::make_pair(low, high);
}

#if H5_VERSION_GE(1, 10, 1)
inline H5F_fspace_strategy_t File::getFileSpaceStrategy() const {
    auto fcpl = getCreatePropertyList();

    H5F_fspace_strategy_t strategy;
    hbool_t persist;
    hsize_t threshold;

    if (H5Pget_file_space_strategy(fcpl.getId(), &strategy, &persist, &threshold) < 0) {
        HDF5ErrMapper::ToException<FileException>(std::string("Unable to get file space strategy"));
    }

    return strategy;
}

inline hsize_t File::getFileSpacePageSize() const {
    auto fcpl = getCreatePropertyList();
    hsize_t page_size;

    if (getFileSpaceStrategy() != H5F_FSPACE_STRATEGY_PAGE) {
        HDF5ErrMapper::ToException<FileException>(
            std::string("Cannot obtain page size as paged allocation is not used."));
    }

    if (H5Pget_file_space_page_size(fcpl.getId(), &page_size) < 0) {
        HDF5ErrMapper::ToException<FileException>(
            std::string("Unable to get file space page size"));
    }

    return page_size;
}
#endif

inline void File::flush() {
    if (H5Fflush(_hid, H5F_SCOPE_GLOBAL) < 0) {
        HDF5ErrMapper::ToException<FileException>(std::string("Unable to flush file " + getName()));
    }
}

}  // namespace HighFive

#endif  // H5FILE_MISC_HPP

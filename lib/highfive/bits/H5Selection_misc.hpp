/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5SELECTION_MISC_HPP
#define H5SELECTION_MISC_HPP

namespace HighFive {

inline Selection::Selection(const DataSpace& memspace,
                            const DataSpace& file_space,
                            const DataSet& set)
    : _mem_space(memspace)
    , _file_space(file_space)
    , _set(set) {}

inline DataSpace Selection::getSpace() const noexcept {
    return _file_space;
}

inline DataSpace Selection::getMemSpace() const noexcept {
    return _mem_space;
}

inline DataSet& Selection::getDataset() noexcept {
    return _set;
}

inline const DataSet& Selection::getDataset() const noexcept {
    return _set;
}

// Not only a shortcut but also for templated compat with H5Dataset
inline const DataType Selection::getDataType() const {
    return _set.getDataType();
}

}  // namespace HighFive

#endif  // H5SELECTION_MISC_HPP

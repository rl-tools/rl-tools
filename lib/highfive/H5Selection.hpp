/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5SELECTION_HPP
#define H5SELECTION_HPP

#include "H5DataSet.hpp"
#include "H5DataSpace.hpp"
#include "bits/H5Slice_traits.hpp"

namespace HighFive {

///
/// \brief Selection: represent a view on a slice/part of a dataset
///
/// A Selection is valid only if its parent dataset is valid
///
class Selection: public SliceTraits<Selection> {
  public:
    ///
    /// \brief getSpace
    /// \return Dataspace associated with this selection
    ///
    DataSpace getSpace() const noexcept;

    ///
    /// \brief getMemSpace
    /// \return Dataspace associated with the memory representation of this
    /// selection
    ///
    DataSpace getMemSpace() const noexcept;

    ///
    /// \brief getDataSet
    /// \return parent dataset of this selection
    ///
    DataSet& getDataset() noexcept;
    const DataSet& getDataset() const noexcept;

    ///
    /// \brief return the datatype of the selection
    /// \return return the datatype of the selection
    const DataType getDataType() const;

  private:
    Selection(const DataSpace& memspace, const DataSpace& file_space, const DataSet& set);

    DataSpace _mem_space, _file_space;
    DataSet _set;

    template <typename Derivate>
    friend class ::HighFive::SliceTraits;
    // absolute namespace naming due to GCC bug 52625
};

}  // namespace HighFive

#endif  // H5SELECTION_HPP

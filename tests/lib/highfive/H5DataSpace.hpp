/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5DATASPACE_HPP
#define H5DATASPACE_HPP

#include <vector>
#include <array>
#include <cstdint>
#include <type_traits>
#include <initializer_list>

#include "H5Object.hpp"
#include "bits/H5_definitions.hpp"

namespace HighFive {

///
/// \brief Class representing the space (dimensions) of a dataset
///
class DataSpace: public Object {
  public:
    const static ObjectType type = ObjectType::DataSpace;

    static const size_t UNLIMITED = SIZE_MAX;

    /// dataspace type
    enum DataspaceType {
        dataspace_scalar,
        dataspace_null
        // simple dataspace are handle directly from their dimensions
    };

    /// create a dataspace of N-dimensions
    /// Each dimension is configured this way
    ///  size(dim1) = vec[0]
    ///  size(dim2) = vec[1]
    ///  etc...
    explicit DataSpace(const std::vector<size_t>& dims);

    // create a dataspace of N-dimensions
    template <size_t N>
    explicit DataSpace(const std::array<size_t, N>& dims);

    /// Make sure that DataSpace({1,2,3}) works on GCC. This is
    /// the shortcut form of the vector initializer, but one some compilers (gcc)
    /// this does not resolve correctly without this constructor.
    DataSpace(const std::initializer_list<size_t>& items);

    /// Allow directly listing 1 or more dimensions to initialize,
    /// that is, DataSpace(1,2) means DataSpace(std::vector<size_t>{1,2}).
    template <typename... Args>
    explicit DataSpace(size_t dim1, Args... dims);

    /// Create a dataspace from an iterator pair
    ///
    /// Explicitly disable DataSpace(int_like, int_like) from trying to use this constructor
    template <typename IT,
              typename = typename std::enable_if<!std::is_integral<IT>::value, IT>::type>
    DataSpace(const IT begin, const IT end);

    /// \brief Create a resizable N-dimensional dataspace
    /// \param dims Initial size of dataspace
    /// \param maxdims Maximum size of the dataspace
    explicit DataSpace(const std::vector<size_t>& dims, const std::vector<size_t>& maxdims);

    ///
    /// \brief DataSpace create a scalar dataspace or a null dataset
    ///
    explicit DataSpace(DataspaceType dtype);

    /// Create a new DataSpace
    ///  with a different id available for modifications
    DataSpace clone() const;

    ///
    /// \brief getNumberDimensions
    /// \return the number of dimensions in the current dataspace
    ///
    size_t getNumberDimensions() const;

    /// \brief getDimensions
    /// \return return a vector of N-element, each element is the size of the
    /// associated dataset dimension
    std::vector<size_t> getDimensions() const;

    /// \brief getElementCount
    /// \return the total number of elements in the dataspace
    size_t getElementCount() const;

    /// \brief getMaxDimensions
    /// \return return a vector of N-element, each element is the size of the
    /// associated dataset maximum dimension
    std::vector<size_t> getMaxDimensions() const;

    /// Create a dataspace matching a type accepted by details::inspector
    template <typename T>
    static DataSpace From(const T& value);

    template <std::size_t N, std::size_t Width>
    static DataSpace FromCharArrayStrings(const char (&)[N][Width]);

  protected:
    DataSpace() = default;

    friend class Attribute;
    friend class File;
    friend class DataSet;
};

}  // namespace HighFive

// We include bits right away since DataSpace is user-constructible
#include "bits/H5Dataspace_misc.hpp"

#endif  // H5DATASPACE_HPP

/*
 *  Copyright (c), 2017, Ali Can Demiralp <ali.demiralp@rwth-aachen.de>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5ATTRIBUTE_HPP
#define H5ATTRIBUTE_HPP

#include <vector>

#include "H5Apublic.h"

#include "H5DataSpace.hpp"
#include "H5DataType.hpp"
#include "H5Object.hpp"
#include "bits/H5Path_traits.hpp"

namespace HighFive {

///
/// \brief Class representing an attribute of a dataset or group
///
class Attribute: public Object, public PathTraits<Attribute> {
  public:
    const static ObjectType type = ObjectType::Attribute;

    ///
    /// \brief return the name of the current attribute
    /// \return the name of the attribute
    std::string getName() const;

    size_t getStorageSize() const;

    ///
    /// \brief getDataType
    /// \return return the datatype associated with this dataset
    ///
    DataType getDataType() const;

    ///
    /// \brief getSpace
    /// \return return the dataspace associated with this dataset
    ///
    DataSpace getSpace() const;

    ///
    /// \brief getMemSpace
    /// \return same than getSpace for DataSet, compatibility with Selection
    /// class
    ///
    DataSpace getMemSpace() const;

    /// \brief Return the attribute
    template <typename T>
    T read() const;

    ///
    /// Read the attribute into a buffer
    /// An exception is raised if the numbers of dimension of the buffer and of
    /// the attribute are different
    ///
    /// The array type can be a N-pointer or a N-vector ( e.g int** integer two
    /// dimensional array )
    template <typename T>
    void read(T& array) const;

    /// \brief Read the attribute into a buffer
    template <typename T>
    void read(T* array, const DataType& dtype = DataType()) const;

    ///
    /// Write the integrality N-dimension buffer to this attribute
    /// An exception is raised if the numbers of dimension of the buffer and of
    /// the attribute are different
    ///
    /// The array type can be a N-pointer or a N-vector ( e.g int** integer two
    /// dimensional array )
    template <typename T>
    void write(const T& buffer);

    /// \brief Write a buffer to this attribute
    template <typename T>
    void write_raw(const T* buffer, const DataType& dtype = DataType());

    /// \brief Get the list of properties for creation of this attribute
    AttributeCreateProps getCreatePropertyList() const {
        return details::get_plist<AttributeCreateProps>(*this, H5Aget_create_plist);
    }

    // No empty attributes
    Attribute() = delete;

  private:
    using Object::Object;

    template <typename Derivate>
    friend class ::HighFive::AnnotateTraits;
};

}  // namespace HighFive


#endif  // H5ATTRIBUTE_HPP

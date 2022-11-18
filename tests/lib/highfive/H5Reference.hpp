/*
 *  Copyright (c), 2020, EPFL - Blue Brain Project
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */

#ifndef H5REFERENCE_HPP
#define H5REFERENCE_HPP

#include <string>
#include <vector>

#include <H5Ipublic.h>
#include <H5Rpublic.h>

#include "H5Object.hpp"
#include "bits/H5_definitions.hpp"

namespace HighFive {

namespace details {
template <typename T>
struct inspector;
}
///
/// \brief An HDF5 (object) reference type
///
/// HDF5 object references allow pointing to groups, datasets (and compound types). They
/// differ from links in their ability to be stored and retrieved as data from the HDF5
/// file in datasets themselves.
///
class Reference {
  public:
    /// \brief Create an empty Reference to be initialized later
    Reference() = default;

    /// \brief Create a Reference to an object residing at a given location
    ///
    /// \param location A File or Group where the object being referenced to resides
    /// \param object A Dataset or Group to be referenced
    Reference(const Object& location, const Object& object);

    /// \brief Retrieve the Object being referenced by the Reference
    ///
    /// \tparam T the appropriate HighFive Container (either DataSet or Group)
    /// \param location the location where the referenced object is to be found (a File)
    /// \return the dereferenced Object (either a Group or DataSet)
    template <typename T>
    T dereference(const Object& location) const;

    /// \brief Get only the type of the referenced Object
    ///
    /// \param location the location where the referenced object is to be found (a File)
    /// \return the ObjectType of the referenced object
    ObjectType getType(const Object& location) const;

  protected:
    /// \brief Create a Reference from a low-level HDF5 object reference
    inline explicit Reference(const hobj_ref_t h5_ref)
        : href(h5_ref){};

    /// \brief Create the low-level reference and store it at refptr
    ///
    /// \param refptr Pointer to a memory location where the created HDF5 reference will
    /// be stored
    void create_ref(hobj_ref_t* refptr) const;

  private:
    Object get_ref(const Object& location) const;

    hobj_ref_t href{};
    std::string obj_name{};
    hid_t parent_id{};

    friend struct details::inspector<Reference>;
};

}  // namespace HighFive

#include "bits/H5Reference_misc.hpp"

#endif  // H5REFERENCE_HPP

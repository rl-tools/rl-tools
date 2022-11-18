/*
 *  Copyright (c), 2020, EPFL - Blue Brain Project
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#pragma once

#include "H5_definitions.hpp"

namespace HighFive {

template <typename Derivate>
class PathTraits {
  public:
    PathTraits();

    ///
    /// \brief return the path to the current object
    /// \return the path to the object
    std::string getPath() const;

    ///
    /// \brief Return a reference to the File object this object belongs
    /// \return the File object ref
    File& getFile() const noexcept;


  protected:
    std::shared_ptr<File> _file_obj;  // keep a ref to file so we keep its ref count > 0
};

}  // namespace HighFive

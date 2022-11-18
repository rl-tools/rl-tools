/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5FILEDRIVER_HPP
#define H5FILEDRIVER_HPP

#include "H5PropertyList.hpp"
#include "bits/H5_definitions.hpp"

namespace HighFive {

///
/// \brief file driver base concept
///
class H5_DEPRECATED("Use FileAccessProps directly") FileDriver: public FileAccessProps {};

#ifdef H5_HAVE_PARALLEL
///
/// \brief MPIIO Driver for Parallel HDF5
///

class H5_DEPRECATED("Add MPIOFileAccess directly to FileAccessProps") MPIOFileDriver
    : public FileAccessProps {
  public:
    inline MPIOFileDriver(MPI_Comm mpi_comm, MPI_Info mpi_info);

  private:
};
#endif

}  // namespace HighFive

#include "bits/H5FileDriver_misc.hpp"

#endif  // H5FILEDRIVER_HPP

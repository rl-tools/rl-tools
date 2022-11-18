/*
 *  Copyright (c), 2017, Blue Brain Project - EPFL (CH)
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */

#ifndef H5UTILITY_HPP
#define H5UTILITY_HPP

#include <H5Epublic.h>

namespace HighFive {

///
/// \brief Utility class to disable HDF5 stack printing inside a scope.
///
class SilenceHDF5 {
  public:
    inline SilenceHDF5(bool enable = true)
        : _client_data(nullptr) {
        H5Eget_auto2(H5E_DEFAULT, &_func, &_client_data);
        if (enable)
            H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
    }

    inline ~SilenceHDF5() {
        H5Eset_auto2(H5E_DEFAULT, _func, _client_data);
    }

  private:
    H5E_auto2_t _func;
    void* _client_data;
};

}  // namespace HighFive

#endif  // H5UTIL_HPP

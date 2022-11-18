/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5EASY_BITS_OPENCV_HPP
#define H5EASY_BITS_OPENCV_HPP

#include "../H5Easy.hpp"
#include "H5Easy_misc.hpp"
#include "H5Easy_scalar.hpp"

#ifdef H5_USE_OPENCV

namespace H5Easy {

namespace detail {

template <class T>
struct is_opencv: std::false_type {};
template <class T>
struct is_opencv<cv::Mat_<T>>: std::true_type {};

template <typename T>
struct io_impl<T, typename std::enable_if<is_opencv<T>::value>::type> {
    inline static std::vector<size_t> shape(const T& data) {
        return std::vector<size_t>{static_cast<size_t>(data.rows), static_cast<size_t>(data.cols)};
    }

    inline static std::vector<int> shape(const File& file,
                                         const std::string& path,
                                         std::vector<size_t> dims) {
        if (dims.size() == 1) {
            return std::vector<int>{static_cast<int>(dims[0]), 1ul};
        }
        if (dims.size() == 2) {
            return std::vector<int>{static_cast<int>(dims[0]), static_cast<int>(dims[1])};
        }

        throw detail::error(file, path, "H5Easy::load: Inconsistent rank");
    }

    inline static DataSet dump(File& file,
                               const std::string& path,
                               const T& data,
                               const DumpOptions& options) {
        using value_type = typename T::value_type;
        DataSet dataset = initDataset<value_type>(file, path, shape(data), options);
        std::vector<value_type> v(data.begin(), data.end());
        dataset.write_raw(v.data());
        if (options.flush()) {
            file.flush();
        }
        return dataset;
    }

    inline static T load(const File& file, const std::string& path) {
        using value_type = typename T::value_type;
        DataSet dataset = file.getDataSet(path);
        std::vector<int> dims = shape(file, path, dataset.getDimensions());
        T data(dims[0], dims[1]);
        dataset.read(reinterpret_cast<value_type*>(data.data));
        return data;
    }

    inline static Attribute dumpAttribute(File& file,
                                          const std::string& path,
                                          const std::string& key,
                                          const T& data,
                                          const DumpOptions& options) {
        using value_type = typename T::value_type;
        Attribute attribute = initAttribute<value_type>(file, path, key, shape(data), options);
        std::vector<value_type> v(data.begin(), data.end());
        attribute.write_raw(v.data());
        if (options.flush()) {
            file.flush();
        }
        return attribute;
    }

    inline static T loadAttribute(const File& file,
                                  const std::string& path,
                                  const std::string& key) {
        using value_type = typename T::value_type;
        DataSet dataset = file.getDataSet(path);
        Attribute attribute = dataset.getAttribute(key);
        DataSpace dataspace = attribute.getSpace();
        std::vector<int> dims = shape(file, path, dataspace.getDimensions());
        T data(dims[0], dims[1]);
        attribute.read(reinterpret_cast<value_type*>(data.data));
        return data;
    }
};

}  // namespace detail
}  // namespace H5Easy

#endif  // H5_USE_OPENCV
#endif  // H5EASY_BITS_OPENCV_HPP

/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5EASY_BITS_SCALAR_HPP
#define H5EASY_BITS_SCALAR_HPP

#include "../H5Easy.hpp"
#include "H5Easy_misc.hpp"

namespace H5Easy {

namespace detail {

/*
Base template for partial specialization: the fallback if specialized templates don't match.
Used e.g. for scalars.
*/
template <typename T, typename = void>
struct io_impl {
    inline static DataSet dump(File& file,
                               const std::string& path,
                               const T& data,
                               const DumpOptions& options) {
        DataSet dataset = initScalarDataset(file, path, data, options);
        dataset.write(data);
        if (options.flush()) {
            file.flush();
        }
        return dataset;
    }

    inline static T load(const File& file, const std::string& path) {
        DataSet dataset = file.getDataSet(path);
        T data;
        dataset.read(data);
        return data;
    }

    inline static Attribute dumpAttribute(File& file,
                                          const std::string& path,
                                          const std::string& key,
                                          const T& data,
                                          const DumpOptions& options) {
        Attribute attribute = initScalarAttribute(file, path, key, data, options);
        attribute.write(data);
        if (options.flush()) {
            file.flush();
        }
        return attribute;
    }

    inline static T loadAttribute(const File& file,
                                  const std::string& path,
                                  const std::string& key) {
        DataSet dataset = file.getDataSet(path);
        Attribute attribute = dataset.getAttribute(key);
        T data;
        attribute.read(data);
        return data;
    }

    inline static DataSet dump_extend(File& file,
                                      const std::string& path,
                                      const T& data,
                                      const std::vector<size_t>& idx,
                                      const DumpOptions& options) {
        std::vector<size_t> ones(idx.size(), 1);

        if (file.exist(path)) {
            DataSet dataset = file.getDataSet(path);
            std::vector<size_t> dims = dataset.getDimensions();
            std::vector<size_t> shape = dims;
            if (dims.size() != idx.size()) {
                throw detail::error(
                    file,
                    path,
                    "H5Easy::dump: Dimension of the index and the existing field do not match");
            }
            for (size_t i = 0; i < dims.size(); ++i) {
                shape[i] = std::max(dims[i], idx[i] + 1);
            }
            if (shape != dims) {
                dataset.resize(shape);
            }
            dataset.select(idx, ones).write(data);
            if (options.flush()) {
                file.flush();
            }
            return dataset;
        }

        std::vector<size_t> shape = idx;
        const size_t unlim = DataSpace::UNLIMITED;
        std::vector<size_t> unlim_shape(idx.size(), unlim);
        std::vector<hsize_t> chunks(idx.size(), 10);
        if (options.isChunked()) {
            chunks = options.getChunkSize();
            if (chunks.size() != idx.size()) {
                throw error(file, path, "H5Easy::dump: Incorrect dimension ChunkSize");
            }
        }
        for (size_t& i: shape) {
            i++;
        }
        DataSpace dataspace = DataSpace(shape, unlim_shape);
        DataSetCreateProps props;
        props.add(Chunking(chunks));
        DataSet dataset = file.createDataSet(path, dataspace, AtomicType<T>(), props, {}, true);
        dataset.select(idx, ones).write(data);
        if (options.flush()) {
            file.flush();
        }
        return dataset;
    }

    inline static T load_part(const File& file,
                              const std::string& path,
                              const std::vector<size_t>& idx) {
        std::vector<size_t> ones(idx.size(), 1);
        DataSet dataset = file.getDataSet(path);
        T data;
        dataset.select(idx, ones).read(data);
        return data;
    }
};

}  // namespace detail
}  // namespace H5Easy

#endif  // H5EASY_BITS_SCALAR_HPP

/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5EASY_BITS_PUBLIC_HPP
#define H5EASY_BITS_PUBLIC_HPP

#include "../H5Easy.hpp"

namespace H5Easy {

inline Compression::Compression(bool enable) {
    if (enable) {
        m_compression_level = 9;
    } else {
        m_compression_level = 0;
    }
}

template <class T>
inline Compression::Compression(T level)
    : m_compression_level(static_cast<unsigned>(level)) {}

inline unsigned Compression::get() const {
    return m_compression_level;
}

inline void DumpOptions::set(DumpMode mode) {
    m_overwrite = static_cast<bool>(mode);
}

inline void DumpOptions::set(Flush mode) {
    m_flush = static_cast<bool>(mode);
}

inline void DumpOptions::set(const Compression& level) {
    m_compression_level = level.get();
}

template <class T, class... Args>
inline void DumpOptions::set(T arg, Args... args) {
    set(arg);
    set(args...);
}

template <class T>
inline void DumpOptions::setChunkSize(const std::vector<T>& shape) {
    m_chunk_size = std::vector<hsize_t>(shape.begin(), shape.end());
}

inline void DumpOptions::setChunkSize(std::initializer_list<size_t> shape) {
    m_chunk_size = std::vector<hsize_t>(shape.begin(), shape.end());
}

inline bool DumpOptions::overwrite() const {
    return m_overwrite;
}

inline bool DumpOptions::flush() const {
    return m_flush;
}

inline bool DumpOptions::compress() const {
    return m_compression_level > 0;
}

inline unsigned DumpOptions::getCompressionLevel() const {
    return m_compression_level;
}

inline bool DumpOptions::isChunked() const {
    return m_chunk_size.size() > 0;
}

inline std::vector<hsize_t> DumpOptions::getChunkSize() const {
    return m_chunk_size;
}

inline size_t getSize(const File& file, const std::string& path) {
    return file.getDataSet(path).getElementCount();
}

inline std::vector<size_t> getShape(const File& file, const std::string& path) {
    return file.getDataSet(path).getDimensions();
}

template <class T>
inline DataSet dump(File& file,
                    const std::string& path,
                    const T& data,
                    const DumpOptions& options) {
    return detail::io_impl<T>::dump(file, path, data, options);
}

template <class T>
inline DataSet dump(File& file, const std::string& path, const T& data, DumpMode mode) {
    return detail::io_impl<T>::dump(file, path, data, DumpOptions(mode));
}

template <class T>
inline DataSet dump(File& file,
                    const std::string& path,
                    const T& data,
                    const std::vector<size_t>& idx,
                    const DumpOptions& options) {
    return detail::io_impl<T>::dump_extend(file, path, data, idx, options);
}

template <class T>
inline DataSet dump(File& file,
                    const std::string& path,
                    const T& data,
                    const std::initializer_list<size_t>& idx,
                    const DumpOptions& options) {
    return detail::io_impl<T>::dump_extend(file, path, data, idx, options);
}

template <class T>
inline DataSet dump(File& file,
                    const std::string& path,
                    const T& data,
                    const std::vector<size_t>& idx) {
    return detail::io_impl<T>::dump_extend(file, path, data, idx, DumpOptions());
}

template <class T>
inline DataSet dump(File& file,
                    const std::string& path,
                    const T& data,
                    const std::initializer_list<size_t>& idx) {
    return detail::io_impl<T>::dump_extend(file, path, data, idx, DumpOptions());
}

template <class T>
inline T load(const File& file, const std::string& path, const std::vector<size_t>& idx) {
    return detail::io_impl<T>::load_part(file, path, idx);
}

template <class T>
inline T load(const File& file, const std::string& path) {
    return detail::io_impl<T>::load(file, path);
}

template <class T>
inline Attribute dumpAttribute(File& file,
                               const std::string& path,
                               const std::string& key,
                               const T& data,
                               DumpMode mode) {
    return detail::io_impl<T>::dumpAttribute(file, path, key, data, DumpOptions(mode));
}

template <class T>
inline Attribute dumpAttribute(File& file,
                               const std::string& path,
                               const std::string& key,
                               const T& data,
                               const DumpOptions& options) {
    return detail::io_impl<T>::dumpAttribute(file, path, key, data, options);
}

template <class T>
inline T loadAttribute(const File& file, const std::string& path, const std::string& key) {
    return detail::io_impl<T>::loadAttribute(file, path, key);
}

}  // namespace H5Easy

#endif  // H5EASY_BITS_MISC_HPP

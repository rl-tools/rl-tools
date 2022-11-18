/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5SLICE_TRAITS_HPP
#define H5SLICE_TRAITS_HPP

#include <cstdlib>
#include <vector>

#include "H5_definitions.hpp"
#include "H5Utils.hpp"

#include "../H5PropertyList.hpp"

namespace HighFive {

class ElementSet {
  public:
    ///
    /// \brief Create a list of points of N-dimension for selection.
    ///
    /// \param list List of continuous coordinates (e.g.: in 2 dimensions space
    /// `ElementSet{1, 2, 3 ,4}` creates points `(1, 2)` and `(3, 4)`).
    explicit ElementSet(std::initializer_list<std::size_t> list);
    ///
    /// \brief Create a list of points of N-dimension for selection.
    ///
    /// \param list List of N-dim points.
    explicit ElementSet(std::initializer_list<std::vector<std::size_t>> list);
    ///
    /// \brief Create a list of points of N-dimension for selection.
    ///
    /// \param element_ids List of continuous coordinates (e.g.: in 2 dimensions space
    /// `ElementSet{1, 2, 3 ,4}` creates points `(1, 2)` and `(3, 4)`).
    explicit ElementSet(const std::vector<std::size_t>& element_ids);
    ///
    /// \brief Create a list of points of N-dimension for selection.
    ///
    /// \param element_ids List of N-dim points.
    explicit ElementSet(const std::vector<std::vector<std::size_t>>& element_ids);

  private:
    std::vector<std::size_t> _ids;

    template <typename Derivate>
    friend class SliceTraits;
};

namespace detail {

template <class To, class From>
inline std::vector<To> convertSizeVector(const std::vector<From>& from) {
    std::vector<To> to(from.size());
    std::copy(from.cbegin(), from.cend(), to.begin());

    return to;
}
}  // namespace detail

inline std::vector<hsize_t> toHDF5SizeVector(const std::vector<size_t>& from) {
    return detail::convertSizeVector<hsize_t>(from);
}

inline std::vector<size_t> toSTLSizeVector(const std::vector<hsize_t>& from) {
    return detail::convertSizeVector<size_t>(from);
}

struct RegularHyperSlab {
    RegularHyperSlab() = default;

    RegularHyperSlab(std::vector<size_t> offset_,
                     std::vector<size_t> count_ = {},
                     std::vector<size_t> stride_ = {},
                     std::vector<size_t> block_ = {})
        : offset(toHDF5SizeVector(offset_))
        , count(toHDF5SizeVector(count_))
        , stride(toHDF5SizeVector(stride_))
        , block(toHDF5SizeVector(block_)) {}

    static RegularHyperSlab fromHDF5Sizes(std::vector<hsize_t> offset_,
                                          std::vector<hsize_t> count_ = {},
                                          std::vector<hsize_t> stride_ = {},
                                          std::vector<hsize_t> block_ = {}) {
        RegularHyperSlab slab;
        slab.offset = offset_;
        slab.count = count_;
        slab.stride = stride_;
        slab.block = block_;

        return slab;
    }

    size_t rank() const {
        return std::max(std::max(offset.size(), count.size()),
                        std::max(stride.size(), block.size()));
    }

    /// Dimensions when all gaps are removed.
    std::vector<size_t> packedDims() const {
        auto n_dims = rank();
        auto dims = std::vector<size_t>(n_dims, 0);

        for (size_t i = 0; i < n_dims; ++i) {
            dims[i] = count[i] * (block.empty() ? 1 : block[i]);
        }

        return dims;
    }

    std::vector<hsize_t> offset;
    std::vector<hsize_t> count;
    std::vector<hsize_t> stride;
    std::vector<hsize_t> block;
};

class HyperSlab {
  public:
    HyperSlab() {
        selects.emplace_back(RegularHyperSlab{}, Op::None);
    };

    explicit HyperSlab(const RegularHyperSlab& sel) {
        selects.emplace_back(sel, Op::Set);
    }

    HyperSlab operator|(const RegularHyperSlab& sel) const {
        auto ret = *this;
        ret |= sel;
        return ret;
    }

    HyperSlab& operator|=(const RegularHyperSlab& sel) {
        selects.emplace_back(sel, Op::Or);
        return *this;
    }

    HyperSlab operator&(const RegularHyperSlab& sel) const {
        auto ret = *this;
        ret &= sel;
        return ret;
    }

    HyperSlab& operator&=(const RegularHyperSlab& sel) {
        selects.emplace_back(sel, Op::And);
        return *this;
    }

    HyperSlab operator^(const RegularHyperSlab& sel) const {
        auto ret = *this;
        ret ^= sel;
        return ret;
    }

    HyperSlab& operator^=(const RegularHyperSlab& sel) {
        selects.emplace_back(sel, Op::Xor);
        return *this;
    }

    HyperSlab& notA(const RegularHyperSlab& sel) {
        selects.emplace_back(sel, Op::NotA);
        return *this;
    }

    HyperSlab& notB(const RegularHyperSlab& sel) {
        selects.emplace_back(sel, Op::NotB);
        return *this;
    }

    DataSpace apply(const DataSpace& space_) const {
        auto space = space_.clone();
        for (const auto& sel: selects) {
            if (sel.op == Op::None) {
                H5Sselect_none(space.getId());
            } else {
                auto error_code =
                    H5Sselect_hyperslab(space.getId(),
                                        convert(sel.op),
                                        sel.offset.empty() ? nullptr : sel.offset.data(),
                                        sel.stride.empty() ? nullptr : sel.stride.data(),
                                        sel.count.empty() ? nullptr : sel.count.data(),
                                        sel.block.empty() ? nullptr : sel.block.data());

                if (error_code < 0) {
                    HDF5ErrMapper::ToException<DataSpaceException>("Unable to select hyperslab");
                }
            }
        }
        return space;
    }

  private:
    enum class Op {
        Noop,
        Set,
        Or,
        And,
        Xor,
        NotB,
        NotA,
        Append,
        Prepend,
        Invalid,
        None,
    };

    H5S_seloper_t convert(Op op) const {
        switch (op) {
        case Op::Noop:
            return H5S_SELECT_NOOP;
        case Op::Set:
            return H5S_SELECT_SET;
        case Op::Or:
            return H5S_SELECT_OR;
        case Op::And:
            return H5S_SELECT_AND;
        case Op::Xor:
            return H5S_SELECT_XOR;
        case Op::NotB:
            return H5S_SELECT_NOTB;
        case Op::NotA:
            return H5S_SELECT_NOTA;
        case Op::Append:
            return H5S_SELECT_APPEND;
        case Op::Prepend:
            return H5S_SELECT_PREPEND;
        case Op::Invalid:
            return H5S_SELECT_INVALID;
        default:
            throw DataSpaceException("Invalid HyperSlab operation.");
        }
    }

    struct Select_: public RegularHyperSlab {
        Select_(const RegularHyperSlab& sel, Op op_)
            : RegularHyperSlab(sel)
            , op(op_) {}

        Op op;
    };

    std::vector<Select_> selects;
};

template <typename Derivate>
class SliceTraits {
  public:
    ///
    /// \brief Select an \p hyperslab in the current Slice/Dataset.
    ///
    /// HyperSlabs can be either regular or irregular. Irregular hyperslabs are typically generated
    /// by taking the union of regular hyperslabs. An irregular hyperslab, in general, does not fit
    /// nicely into a multi-dimensional array, but only a subset of such an array.
    ///
    /// Therefore, the only memspaces supported for general hyperslabs are one-dimensional arrays.
    Selection select(const HyperSlab& hyperslab) const;

    ///
    /// \brief Select a region in the current Slice/Dataset of \p count points at
    /// \p offset separated by \p stride. If strides are not provided they will
    /// default to 1 in all dimensions.
    ///
    /// vector offset and count have to be from the same dimension
    ///
    Selection select(const std::vector<size_t>& offset,
                     const std::vector<size_t>& count,
                     const std::vector<size_t>& stride = {},
                     const std::vector<size_t>& block = {}) const;

    ///
    /// \brief Select a set of columns in the last dimension of this dataset.
    ///
    /// The column indices must be smaller than the dimension size.
    ///
    Selection select(const std::vector<size_t>& columns) const;

    ///
    /// \brief Select a region in the current Slice/Dataset out of a list of elements.
    ///
    Selection select(const ElementSet& elements) const;

    template <typename T>
    T read(const DataTransferProps& xfer_props = DataTransferProps()) const;

    ///
    /// Read the entire dataset into a buffer
    /// An exception is raised is if the numbers of dimension of the buffer and
    /// of the dataset are different.
    ///
    /// The array type can be a N-pointer or a N-vector. For plain pointers
    /// not dimensionality checking will be performed, it is the user's
    /// responsibility to ensure that the right amount of space has been
    /// allocated.
    template <typename T>
    void read(T& array, const DataTransferProps& xfer_props = DataTransferProps()) const;

    ///
    /// Read the entire dataset into a raw buffer
    ///
    /// No dimensionality checks will be performed, it is the user's
    /// responsibility to ensure that the right amount of space has been
    /// allocated.
    /// \param array: A buffer containing enough space for the data
    /// \param dtype: The type of the data, in case it cannot be automatically guessed
    template <typename T>
    void read(T* array,
              const DataType& dtype = DataType(),
              const DataTransferProps& xfer_props = DataTransferProps()) const;

    ///
    /// Write the integrality N-dimension buffer to this dataset
    /// An exception is raised is if the numbers of dimension of the buffer and
    /// of the dataset are different
    ///
    /// The array type can be a N-pointer or a N-vector ( e.g int** integer two
    /// dimensional array )
    template <typename T>
    void write(const T& buffer, const DataTransferProps& xfer_props = DataTransferProps());

    ///
    /// Write from a raw buffer into this dataset
    ///
    /// No dimensionality checks will be performed, it is the user's
    /// responsibility to ensure that the buffer holds the right amount of
    /// elements. For n-dimensional matrices the buffer layout follows H5
    /// default conventions.
    /// \param buffer: A buffer containing the data to be written
    /// \param dtype: The type of the data, in case it cannot be automatically guessed
    /// \param xfer_props: The HDF5 data transfer properties, e.g. collective MPI-IO.
    template <typename T>
    void write_raw(const T* buffer,
                   const DataType& dtype = DataType(),
                   const DataTransferProps& xfer_props = DataTransferProps());

  protected:
    inline Selection select_impl(const HyperSlab& hyperslab, const DataSpace& memspace) const;
};

}  // namespace HighFive

#endif  // H5SLICE_TRAITS_HPP

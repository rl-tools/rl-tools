#include

namespace layer_in_c::utils{
    template <typename T, index_t ROWS, index_t COLS>
    using matrix = std::array<std::array<T, COLS>, ROWS>;
}
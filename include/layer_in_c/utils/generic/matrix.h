#include

namespace layer_in_c::utils{
    template <typename T, size_t ROWS, size_t COLS>
    using matrix = std::array<std::array<T, COLS>, ROWS>;
}
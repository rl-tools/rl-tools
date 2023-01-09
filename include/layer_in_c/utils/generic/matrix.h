#include

namespace layer_in_c::utils{
    template <typename T, auto ROWS, auto COLS>
    using matrix = std::array<std::array<T, COLS>, ROWS>;
}
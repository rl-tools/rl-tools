#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UTILS_GENERIC_TUPLE_TUPLE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UTILS_GENERIC_TUPLE_TUPLE_H

#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace utils{
        template<typename T_TI, typename... Types>
        struct Tuple {
            using TI = T_TI;
        };

        template<typename T_TI>
        struct Tuple<T_TI> {
        };

        template<typename T_TI, typename Head, typename... Tail>
        struct Tuple<T_TI, Head, Tail...> : Tuple<T_TI, Tail...> {
            using NEXT = Tuple<T_TI, Tail...>;
            Head content;
        };

        template<typename TUPLE, template<typename> typename F>
        struct MapTuple;

        template<typename T_TI, template<typename> typename F>
        struct MapTuple<Tuple<T_TI>, F> {
            using type = Tuple<T_TI>;
        };


        template<typename T_TI, typename Head, typename... Tail, template<typename> typename F>
        struct MapTuple<Tuple<T_TI, Head, Tail...>, F> : MapTuple<Tuple<T_TI, Tail...>, F> {
            using CONTENT = typename F<Head>::CONTENT;
            CONTENT content;
        };

        template <typename TUPLE>
        struct tuple_size;

        template <typename TI, typename... TYPES>
        struct tuple_size<Tuple<TI, TYPES...>> {
            static constexpr TI value = sizeof...(TYPES);
        };

        template <typename TUPLE, typename T>
        struct tuple_append;

        template <typename TI, typename... TYPES, typename T>
        struct tuple_append<Tuple<TI, TYPES...>, T> {
            using type = Tuple<TI, TYPES..., T>;
        };

        template <typename TUPLE, typename T>
        using tuple_append_t = typename tuple_append<TUPLE, T>::type;

        template <auto INDEX, typename TUPLE>
        struct tuple_element;

        namespace detail {
            template <typename TI, TI... Is>
            struct index_sequence {};

            template <typename TI, bool DONE, TI N, TI... Is>
            struct make_index_sequence_impl;
            template <typename TI, TI N, TI... Is>
            struct make_index_sequence_impl<TI, true, N, Is...> {
                using type = index_sequence<TI, Is...>;
            };
            template <typename TI, TI N, TI... Is>
            struct make_index_sequence_impl<TI, false, N, Is...> {
                using type = typename make_index_sequence_impl<TI, N - 1 == 0, N - 1, N - 1, Is...>::type;
            };
            template <typename TI, TI N>
            using make_index_sequence = typename make_index_sequence_impl<TI, N == 0, N>::type;

            template <typename TI, TI Index, typename T>
            struct TupleLeaf {
                using type = T;
            };

            template <typename SEQ, typename TI, typename... Ts>
            struct TupleIndex;
            template <typename TI, TI... Is, typename... Ts>
            struct TupleIndex<index_sequence<TI, Is...>, TI, Ts...> : TupleLeaf<TI, Is, Ts>... {};

            template <typename TI, TI I, typename T>
            TupleLeaf<TI, I, T> select_leaf(const TupleLeaf<TI, I, T>&);
        }

        template <auto INDEX, typename TI, typename... TYPES>
        struct tuple_element<INDEX, Tuple<TI, TYPES...>> {
            static_assert(static_cast<TI>(INDEX) < sizeof...(TYPES), "tuple_element index out of bounds");
            using Indexed = detail::TupleIndex<detail::make_index_sequence<TI, sizeof...(TYPES)>, TI, TYPES...>;
            using type = typename decltype(detail::select_leaf<TI, static_cast<TI>(INDEX)>(Indexed{}))::type;
        };
    }
    template<typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr TI length(utils::Tuple<TI> &tuple) {
        return 0;
    }

    template<typename TI, typename CURRENT_TYPE, typename... Types>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr TI length(utils::Tuple<TI, CURRENT_TYPE, Types...> &tuple) {
        return 1 + length(static_cast<utils::Tuple<TI, Types...> &>(tuple));
    }

    template<auto INDEX, typename TI, typename CURRENT_TYPE, typename... Types>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr auto &get(utils::Tuple<TI, CURRENT_TYPE, Types...> &tuple) {
        if constexpr (INDEX == 0) {
            return tuple.content;
        } else {
            return get<INDEX - 1>(static_cast<utils::Tuple<TI, Types...> &>(tuple));
        }
    }
    template<auto INDEX, typename TI, typename CURRENT_TYPE, typename... Types>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr const auto &get(const utils::Tuple<TI, CURRENT_TYPE, Types...> &tuple) {
        if constexpr (INDEX == 0) {
            return tuple.content;
        } else {
            return get<INDEX - 1>(static_cast<const utils::Tuple<TI, Types...> &>(tuple));
        }
    }

    template<auto INDEX, typename TI, typename CURRENT_TYPE, typename... Types, template <typename> typename F>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr auto &get(utils::MapTuple<utils::Tuple<TI, CURRENT_TYPE, Types...>, F> &tuple) {
        if constexpr (INDEX == 0) {
            return tuple.content;
        } else {
            return get<INDEX - 1>(static_cast<utils::MapTuple<utils::Tuple<TI, Types...>, F> &>(tuple));
        }
    }
    template<auto INDEX, typename TI, typename CURRENT_TYPE, typename... Types, template <typename> typename F>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr const auto &get(const utils::MapTuple<utils::Tuple<TI, CURRENT_TYPE, Types...>, F> &tuple) {
        if constexpr (INDEX == 0) {
            return tuple.content;
        } else {
            return get<INDEX - 1>(static_cast<const utils::MapTuple<utils::Tuple<TI, Types...>, F> &>(tuple));
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif

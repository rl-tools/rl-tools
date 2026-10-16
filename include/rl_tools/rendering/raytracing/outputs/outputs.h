#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_OUTPUTS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUTS_H

#include "../types.h"
#include "../../../containers/tensor/tensor.h"
#include "../../../utils/generic/tuple/tuple.h"
#include "../../../utils/generic/typing.h"
#include "slots.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::raytracing::outputs {
    namespace format {
        struct RGBA8 { using T = unsigned int; };
        struct Float32 { using T = float; };
        struct UInt32 { using T = unsigned int; };
    }
    namespace labels {
        struct Instance {};
        struct Semantic {};
    }
    namespace direction {
        struct Backward {};
        struct Forward {};
    }
    struct DepthParameters { static constexpr float SCALE = 10; };
    struct FlowParameters { static constexpr float SCALE = 20; };

    template <typename T_FORMAT = format::RGBA8>
    struct RGB { using FORMAT = T_FORMAT; };
    template <typename T_FORMAT = format::Float32, typename T_PARAMETERS = DepthParameters>
    struct Depth { using FORMAT = T_FORMAT; using PARAMETERS = T_PARAMETERS; };
    template <typename T_FORMAT = format::UInt32, typename T_LABELS = labels::Instance>
    struct Segmentation { using FORMAT = T_FORMAT; using LABELS = T_LABELS; };
    template <typename T_FORMAT = format::Float32>
    struct Normals { using FORMAT = T_FORMAT; };
    template <typename T_DIRECTION = direction::Backward, typename T_FORMAT = format::Float32, typename T_PARAMETERS = FlowParameters>
    struct Flow { using DIRECTION = T_DIRECTION; using FORMAT = T_FORMAT; using PARAMETERS = T_PARAMETERS; };
    template <typename T_DIRECTION = direction::Backward, typename T_FORMAT = format::UInt32>
    struct FlowStatus { using DIRECTION = T_DIRECTION; using FORMAT = T_FORMAT; };
    struct Probes {};

    enum class Source { RGB, Surface, BackwardFlow, ForwardFlow, Probe };

    template <typename OUTPUT>
    struct Traits;
    template <typename F>
    struct Traits<RGB<F>> {
        static_assert(utils::typing::is_same_v<F, format::RGBA8> || utils::typing::is_same_v<F, format::Float32>, "unsupported RGB representation");
        using T = typename F::T;
        static constexpr bool PACKED = utils::typing::is_same_v<F, format::RGBA8>;
        static constexpr Slot SLOT = PACKED ? Slot::RGB_RGBA8 : Slot::RGB_FLOAT32;
        static constexpr unsigned int CHANNELS = PACKED ? 1 : 3;
        static constexpr Source SOURCE = Source::RGB;
    };
    template <typename F, typename P>
    struct Traits<Depth<F, P>> {
        static_assert(utils::typing::is_same_v<F, format::Float32> || utils::typing::is_same_v<F, format::RGBA8>, "unsupported depth representation");
        static_assert(P::SCALE > 0, "depth display scale must be positive");
        using T = typename F::T;
        static constexpr Slot SLOT = utils::typing::is_same_v<F, format::RGBA8> ? Slot::DEPTH_RGBA8 : Slot::DEPTH_FLOAT32;
        static constexpr unsigned int CHANNELS = 1;
        static constexpr Source SOURCE = Source::Surface;
    };
    template <typename F, typename L>
    struct Traits<Segmentation<F, L>> {
        static_assert(utils::typing::is_same_v<F, format::UInt32> || utils::typing::is_same_v<F, format::RGBA8>, "unsupported segmentation representation");
        static_assert(utils::typing::is_same_v<L, labels::Instance> || utils::typing::is_same_v<L, labels::Semantic>, "unsupported segmentation labels");
        using T = typename F::T;
        static constexpr bool PACKED = utils::typing::is_same_v<F, format::RGBA8>;
        static constexpr bool SEMANTIC = utils::typing::is_same_v<L, labels::Semantic>;
        static constexpr Slot SLOT = SEMANTIC ? (PACKED ? Slot::SEMANTIC_RGBA8 : Slot::SEMANTIC_UINT32) : (PACKED ? Slot::SEGMENTATION_RGBA8 : Slot::SEGMENTATION_UINT32);
        static constexpr unsigned int CHANNELS = 1;
        static constexpr Source SOURCE = Source::Surface;
    };
    template <typename F>
    struct Traits<Normals<F>> {
        static_assert(utils::typing::is_same_v<F, format::Float32> || utils::typing::is_same_v<F, format::RGBA8>, "unsupported normals representation");
        using T = typename F::T;
        static constexpr bool PACKED = utils::typing::is_same_v<F, format::RGBA8>;
        static constexpr Slot SLOT = PACKED ? Slot::NORMALS_RGBA8 : Slot::NORMALS_FLOAT32;
        static constexpr unsigned int CHANNELS = PACKED ? 1 : 3;
        static constexpr Source SOURCE = Source::Surface;
    };
    template <typename D, typename F, typename P>
    struct Traits<Flow<D, F, P>> {
        static_assert(utils::typing::is_same_v<F, format::Float32> || utils::typing::is_same_v<F, format::RGBA8>, "unsupported flow representation");
        static_assert(utils::typing::is_same_v<D, direction::Backward> || utils::typing::is_same_v<D, direction::Forward>, "unsupported flow direction");
        using T = typename F::T;
        static constexpr bool PACKED = utils::typing::is_same_v<F, format::RGBA8>;
        static constexpr bool FORWARD = utils::typing::is_same_v<D, direction::Forward>;
        static constexpr Slot SLOT = FORWARD ? (PACKED ? Slot::FLOW_FORWARD_RGBA8 : Slot::FLOW_FORWARD_FLOAT32) : (PACKED ? Slot::FLOW_RGBA8 : Slot::FLOW_FLOAT32);
        static constexpr unsigned int CHANNELS = PACKED ? 1 : 2;
        static constexpr Source SOURCE = FORWARD ? Source::ForwardFlow : Source::BackwardFlow;
    };
    template <typename D, typename F>
    struct Traits<FlowStatus<D, F>> {
        static_assert(utils::typing::is_same_v<F, format::UInt32> || utils::typing::is_same_v<F, format::RGBA8>, "unsupported flow status representation");
        static_assert(utils::typing::is_same_v<D, direction::Backward> || utils::typing::is_same_v<D, direction::Forward>, "unsupported flow direction");
        using T = typename F::T;
        static constexpr bool PACKED = utils::typing::is_same_v<F, format::RGBA8>;
        static constexpr bool FORWARD = utils::typing::is_same_v<D, direction::Forward>;
        static constexpr Slot SLOT = FORWARD ? (PACKED ? Slot::FLOW_FORWARD_STATUS_RGBA8 : Slot::FLOW_FORWARD_STATUS_UINT32) : (PACKED ? Slot::FLOW_STATUS_RGBA8 : Slot::FLOW_STATUS_UINT32);
        static constexpr unsigned int CHANNELS = 1;
        static constexpr Source SOURCE = FORWARD ? Source::ForwardFlow : Source::BackwardFlow;
    };
    template <>
    struct Traits<Probes> {
        using T = CollisionResult;
        static constexpr Slot SLOT = Slot::PROBES;
        static constexpr unsigned int CHANNELS = 1;
        static constexpr Source SOURCE = Source::Probe;
    };

    // which cameras produce an output: a list entry Cameras<SELECTION, OUTPUT> covers the selected
    // cameras (one tensor row each, in camera order), a bare descriptor covers every camera. The
    // renderer runs each piece of shared per-pixel work on the union of the cameras whose outputs
    // need it. A selection is any type with a constexpr selected(camera)
    namespace cameras {
        struct All {
            template <typename TI>
            RL_TOOLS_FUNCTION_PLACEMENT static constexpr bool selected(TI) { return true; }
        };
        // every PERIOD-th camera at the given offsets, e.g. one mount of every camera rig
        template <auto PERIOD, auto... OFFSETS>
        struct Every {
            static_assert(PERIOD > 0 && sizeof...(OFFSETS) > 0 && ((OFFSETS < PERIOD) && ...), "Every<PERIOD, OFFSETS...> needs offsets below PERIOD");
            template <typename TI>
            RL_TOOLS_FUNCTION_PLACEMENT static constexpr bool selected(TI camera) { return ((camera % (TI)PERIOD == (TI)OFFSETS) || ...); }
        };
        template <typename SELECTION, typename TI>
        constexpr TI count(TI num_cameras) {
            TI selected = 0;
            for(TI camera = 0; camera < num_cameras; camera++) {
                selected += SELECTION::selected(camera) ? 1 : 0;
            }
            return selected;
        }
    }
    template <typename T_SELECTION, typename T_OUTPUT>
    struct Cameras {
        using SELECTION = T_SELECTION;
        using OUTPUT = T_OUTPUT;
    };
    template <typename ENTRY>
    struct Entry {
        using OUTPUT = ENTRY;
        using SELECTION = cameras::All;
    };
    template <typename T_SELECTION, typename T_OUTPUT>
    struct Entry<Cameras<T_SELECTION, T_OUTPUT>> {
        using OUTPUT = T_OUTPUT;
        using SELECTION = T_SELECTION;
    };
    template <typename SELECTION, typename OUTPUT>
    struct Traits<Cameras<SELECTION, OUTPUT>>: Traits<OUTPUT> {};

    template <typename... OUTPUTS>
    struct List;
    template <typename LIST, typename OUTPUT>
    struct contains;
    template <typename... OUTPUTS, typename OUTPUT>
    struct contains<List<OUTPUTS...>, OUTPUT>: utils::typing::integral_constant<bool, (utils::typing::is_same_v<OUTPUT, typename Entry<OUTPUTS>::OUTPUT> || ...)> {};
    // the configured entry (with its camera selection) of a descriptor
    template <typename LIST, typename OUTPUT>
    struct EntryOf;
    template <typename OUTPUT>
    struct EntryOf<List<>, OUTPUT> {
        static_assert(!utils::typing::is_same_v<OUTPUT, OUTPUT>, "output is not configured");
    };
    template <typename ENTRY>
    struct Found {
        using TYPE = ENTRY;
    };
    template <typename HEAD, typename... TAIL, typename OUTPUT>
    struct EntryOf<List<HEAD, TAIL...>, OUTPUT>: utils::typing::conditional_t<utils::typing::is_same_v<typename Entry<HEAD>::OUTPUT, OUTPUT>, Found<HEAD>, EntryOf<List<TAIL...>, OUTPUT>> {};
    template <typename LIST, Slot SLOT>
    struct contains_slot;
    template <typename... OUTPUTS, Slot SLOT>
    struct contains_slot<List<OUTPUTS...>, SLOT>: utils::typing::integral_constant<bool, ((Traits<OUTPUTS>::SLOT == SLOT) || ...)> {};
    template <typename... OUTPUTS>
    struct Unique: utils::typing::true_type {};
    template <typename HEAD, typename... TAIL>
    struct Unique<HEAD, TAIL...>: utils::typing::integral_constant<bool, !contains_slot<List<TAIL...>, Traits<HEAD>::SLOT>::value && Unique<TAIL...>::value> {};
    template <typename... OUTPUTS>
    struct List {
        static_assert(Unique<OUTPUTS...>::value, "configure at most one output per quantity and representation");
        static constexpr bool EMPTY = sizeof...(OUTPUTS) == 0;
        static constexpr unsigned int COUNT = sizeof...(OUTPUTS);
    };
    template <typename... LISTS>
    struct Concat;
    template <>
    struct Concat<> { using type = List<>; };
    template <typename... OUTPUTS>
    struct Concat<List<OUTPUTS...>> { using type = List<OUTPUTS...>; };
    template <typename... A, typename... B, typename... TAIL>
    struct Concat<List<A...>, List<B...>, TAIL...>: Concat<List<A..., B...>, TAIL...> {};
    template <bool ENABLED, typename OUTPUT>
    using If = utils::typing::conditional_t<ENABLED, List<OUTPUT>, List<>>;

    template <typename LIST>
    struct Requirements;
    template <typename... OUTPUTS>
    struct Requirements<List<OUTPUTS...>> {
        template <Slot SLOT>
        static constexpr bool HAS = contains_slot<List<OUTPUTS...>, SLOT>::value;
        static constexpr unsigned int MASK = (0u | ... | (1u << (unsigned int)Traits<OUTPUTS>::SLOT));
        static constexpr bool HAS_RGBA8 = HAS<Slot::RGB_RGBA8>;
        static constexpr bool HAS_RGB32F = HAS<Slot::RGB_FLOAT32>;
        static constexpr bool HAS_RGB = HAS_RGBA8 || HAS_RGB32F;
        static constexpr bool HAS_DEPTH = HAS<Slot::DEPTH_FLOAT32> || HAS<Slot::DEPTH_RGBA8>;
        static constexpr bool HAS_INSTANCE_SEGMENTATION = HAS<Slot::SEGMENTATION_UINT32> || HAS<Slot::SEGMENTATION_RGBA8>;
        static constexpr bool HAS_SEMANTIC_SEGMENTATION = HAS<Slot::SEMANTIC_UINT32> || HAS<Slot::SEMANTIC_RGBA8>;
        static constexpr bool HAS_SEGMENTATION = HAS_INSTANCE_SEGMENTATION || HAS_SEMANTIC_SEGMENTATION;
        static constexpr bool HAS_NORMALS = HAS<Slot::NORMALS_FLOAT32> || HAS<Slot::NORMALS_RGBA8>;
        static constexpr bool HAS_FLOW_BACKWARD = ((Traits<OUTPUTS>::SOURCE == Source::BackwardFlow) || ...);
        static constexpr bool HAS_FLOW_FORWARD = ((Traits<OUTPUTS>::SOURCE == Source::ForwardFlow) || ...);
        static constexpr bool HAS_FLOW_BACKWARD_STATUS = HAS<Slot::FLOW_RGBA8> || HAS<Slot::FLOW_STATUS_UINT32> || HAS<Slot::FLOW_STATUS_RGBA8>;
        static constexpr bool HAS_FLOW_FORWARD_STATUS = HAS<Slot::FLOW_FORWARD_RGBA8> || HAS<Slot::FLOW_FORWARD_STATUS_UINT32> || HAS<Slot::FLOW_FORWARD_STATUS_RGBA8>;
        static constexpr bool HAS_FLOW = HAS_FLOW_BACKWARD || HAS_FLOW_FORWARD;
        static constexpr bool HAS_PROBES = HAS<Slot::PROBES>;
        static constexpr bool HAS_AUXILIARY = HAS_DEPTH || HAS_SEGMENTATION || HAS_NORMALS || HAS_FLOW_BACKWARD;
        static constexpr bool HAS_IMAGE = HAS_RGB || HAS_AUXILIARY || HAS_FLOW_FORWARD;
    };


    template <typename SPEC, typename OUTPUT, typename SPEC::TI ROWS, unsigned int CHANNELS = Traits<OUTPUT>::CHANNELS>
    struct Shape { using TYPE = tensor::Shape<typename SPEC::TI, ROWS, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH, CHANNELS>; };
    template <typename SPEC, typename OUTPUT, typename SPEC::TI ROWS>
    struct Shape<SPEC, OUTPUT, ROWS, 1> { using TYPE = tensor::Shape<typename SPEC::TI, ROWS, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH>; };
    template <typename SPEC, typename SPEC::TI ROWS>
    struct Shape<SPEC, Probes, ROWS, 1> { using TYPE = tensor::Shape<typename SPEC::TI, ROWS, SPEC::NUM_PROBES>; };
    template <typename SPEC, typename ENTRY>
    struct Buffer {
        using OUTPUT = typename Entry<ENTRY>::OUTPUT;
        using SELECTION = typename Entry<ENTRY>::SELECTION;
        static constexpr typename SPEC::TI ROWS = cameras::count<SELECTION>(SPEC::NUM_CAMERAS);
        static_assert(ROWS > 0, "the camera selection of an output selects no camera");
        using TENSOR_SPEC = tensor::Specification<typename Traits<OUTPUT>::T, typename SPEC::TI, typename Shape<SPEC, OUTPUT, ROWS>::TYPE, true>;
        Tensor<TENSOR_SPEC> tensor;
    };
    template <typename SPEC, typename OUTPUT>
    using TensorSpecification = typename Buffer<SPEC, typename EntryOf<typename SPEC::OUTPUTS, OUTPUT>::TYPE>::TENSOR_SPEC;
    template <typename SPEC, typename LIST>
    struct Buffers;
    template <typename SPEC, typename... OUTPUTS>
    struct Buffers<SPEC, List<OUTPUTS...>> {
        static_assert(Unique<OUTPUTS...>::value, "duplicate output quantity and representation");
        utils::Tuple<typename SPEC::TI, Buffer<SPEC, OUTPUTS>...> content;
    };
    struct Offsets {
        unsigned int values[(unsigned int)Slot::COUNT] = {};
        constexpr unsigned int operator[](unsigned int index) const { return values[index]; }
    };
    template <typename SPEC, typename LIST = typename SPEC::OUTPUTS>
    struct Layout;
    template <typename SPEC, typename... OUTPUTS>
    struct Layout<SPEC, List<OUTPUTS...>> {
        static constexpr bool ALL_CAMERAS = ((Buffer<SPEC, OUTPUTS>::ROWS == SPEC::NUM_CAMERAS) && ...);
        static constexpr unsigned int SIZE_BYTES = (0u + ... + Buffer<SPEC, OUTPUTS>::TENSOR_SPEC::SIZE_BYTES);
        static constexpr unsigned int WORDS = SIZE_BYTES / sizeof(unsigned int);
        static constexpr Offsets SLOT_OFFSETS = []() {
            Offsets result;
            for(unsigned int slot = 0; slot < (unsigned int)Slot::COUNT; slot++) {
                result.values[slot] = 0xffffffffu;
            }
            unsigned int cursor = 0;
            ((result.values[(unsigned int)Traits<OUTPUTS>::SLOT] = cursor, cursor += Buffer<SPEC, OUTPUTS>::TENSOR_SPEC::SIZE_BYTES / sizeof(unsigned int)), ...);
            return result;
        }();
        template <typename OUTPUT>
        static constexpr unsigned int offset() {
            static_assert(contains<List<OUTPUTS...>, OUTPUT>::value, "output is not configured");
            return SLOT_OFFSETS[(unsigned int)Traits<OUTPUT>::SLOT] * sizeof(unsigned int);
        }
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif

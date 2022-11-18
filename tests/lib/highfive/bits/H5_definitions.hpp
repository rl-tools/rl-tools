#pragma once

#if defined(__GNUC__) || defined(__clang__)
#define H5_DEPRECATED(msg) __attribute__((deprecated(#msg)))
#elif defined(_MSC_VER)
#define H5_DEPRECATED(msg) __declspec(deprecated(#msg))
#else
#pragma message("WARNING: Compiler doesnt support deprecation")
#define H5_DEPRECATED
#endif


// Forward declarations

namespace HighFive {

enum class LinkType;
enum class ObjectType;
enum class PropertyType;

class Attribute;
class DataSet;
class DataSpace;
class DataType;
class Exception;
class File;
class FileDriver;
class Group;
class Object;
class ObjectInfo;
class Reference;
class Selection;
class SilenceHDF5;

template <typename T>
class AtomicType;

template <typename Derivate>
class AnnotateTraits;

template <std::size_t N>
class FixedLenStringArray;

template <typename Derivate>
class NodeTraits;

template <PropertyType T>
class PropertyList;

}  // namespace HighFive

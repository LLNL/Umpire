#ifndef UMPIRE_MemoryResourceTypes_HPP
#define UMPIRE_MemoryResourceTypes_HPP

namespace umpire {
namespace resource {

struct MemoryResourceTypeHash
{
    template <typename T>
    std::size_t operator()(T t) const
    {
        return static_cast<std::size_t>(t);
    }
};


enum MemoryResourceType {
  Host,
  Device,
  UnifiedMemory,
  PinnedMemory
};

} // end of namespace resource
} // end of namespace umpire

#endif

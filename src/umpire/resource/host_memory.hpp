#pragma once

#include "umpire/resource/memory_resource.hpp"
#include "umpire/alloc/malloc_allocator.hpp"

namespace umpire {
namespace resource {

template<typename Allocator=umpire::alloc::malloc_allocator, 
         bool Tracking=true>
class host_memory :
  public memory_resource<umpire::resource::host_platform>
{
  public:
  static host_memory* get() {
    static host_memory self;
    return &self;
  }

  void* allocate(std::size_t n) final
  {
    if constexpr(Tracking) {
      return this->track_allocation(this, Allocator::allocate(n), n);
    } else {
      return Allocator::allocate(n);
    }
  }

  void deallocate(void* ptr) final
  {
    if constexpr(Tracking) {
      this->untrack_allocation(ptr);
    }
    Allocator::deallocate(ptr);
  }

  camp::resources::Platform get_platform()
  {
    return camp::resources::Platform::host;
  }

  ~host_memory() = default;
  host_memory(const host_memory&) = delete;
  host_memory& operator=(const host_memory&) = delete;

  private:

  host_memory() :
    memory_resource<umpire::resource::host_platform>{"HOST"}
  {}
};

}
}
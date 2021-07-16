#pragma once

#include "umpire/memory.hpp"
#include "umpire/detail/trackable.hpp"

namespace umpire {

template<class T, typename Memory=memory>
class allocator {
  public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using value_type = T;
    using platform = typename Memory::platform;

    allocator(Memory* strategy) : memory_{strategy}{}
    allocator(allocator const& o) : memory_{o.memory_} {}

    allocator& operator=(allocator const&) = default;
    bool operator==(allocator const& o) const{return memory_ == o.memory_;}
    bool operator!=(allocator const& o) const{return memory_ != o.memory_;}

    pointer allocate(size_type n){
      return static_cast<pointer>(memory_->allocate(n*sizeof(value_type)));
    }

    void deallocate(pointer ptr, std::size_t n = static_cast<std::size_t>(-1)){
      if (ptr) {
        (void) n;
        memory_->deallocate(ptr);
      }
    }

    std::size_t
    get_current_size() const noexcept
    {
      return memory_->get_current_size()/sizeof(T);
    }

    std::size_t
    get_actual_size() const noexcept
    {
      return memory_->get_actual_size()/sizeof(T);
    }

    std::size_t
    get_highwatermark() const noexcept
    {
      return memory_->get_highwatermark()/sizeof(T);
    }

    std::string get_name() const noexcept
    {
      return memory_->get_name();
    }

    camp::resources::Platform get_platform() noexcept
    {
      return memory_->get_platform();
    }

    memory* get_memory() const noexcept
    {
      return memory_;
    }

  private:
    Memory* memory_;
};

using Allocator = allocator<char>;

} // end of namespace umpire

template<typename T, typename Memory>
std::ostream& operator<<(std::ostream& os, const umpire::allocator<T, Memory>& a)
{
  os << a.get_name();
  return os;
}
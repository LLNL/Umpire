#pragma once

#include "umpire/allocation_record.hpp"
#include "umpire/detail/registry.hpp"
#include "umpire/detail/trackable.hpp"

namespace umpire {
namespace tracking {

struct default_tracker :
  public trackable
{
    std::size_t get_current_size() const noexcept final
    {
      return current_size_;
    }

    std::size_t get_actual_size() const noexcept final
    {
      return actual_size_;
    }

    std::size_t get_highwatermark() const noexcept final
    {
      return highwatermark_;
    }

  protected:
    template<typename Memory>
    void* track(Memory* self, void* ptr, std::size_t n)
    {
      current_size_ += n;
      actual_size_ += n;
      highwatermark_ = (current_size_ > highwatermark_) 
        ? current_size_ : highwatermark_;

      auto& map = umpire::detail::registry::get()->get_allocation_map();
      map.insert(ptr, umpire::allocation_record{ptr, n, self});
      return ptr;
    }

    void untrack(void* ptr)
    {
      auto& map = umpire::detail::registry::get()->get_allocation_map();
      auto record = map.remove(ptr);
      current_size_ -= record.size;
      actual_size_ -= record.size;
    }

    std::size_t current_size_{0};
    std::size_t actual_size_{0};
    std::size_t highwatermark_{0};
};

struct no_op_tracker :
  public trackable
{
    template<typename Memory>
    constexpr void* track(Memory*, void* ptr, std::size_t)
    {
      return ptr;
    }

    constexpr void untrack(void*)
    {
    }

    std::size_t get_current_size() const noexcept final
    {
      return 0;
    }

    std::size_t get_actual_size() const noexcept final
    {
      return 0;
    }

    std::size_t get_highwatermark() const noexcept final
    {
      return 0;
    }
};

}
}
#pragma once

#include "umpire/resource/platform.hpp"
#include "umpire/detail/registry.hpp"

#include <string>

#include "camp/resource/platform.hpp"

namespace umpire {

class memory
{
  public:
  using platform=umpire::resource::undefined_platform;
  using zero_byte_pool_t = strategy::fixed_pool<resource::null_resource<>>;

  memory(const std::string& name) :
    zero_pool{
      static_cast<zero_byte_pool_t>(detail::registry::get()::get}
    m_id{detail::registry::get()->get_id()},
    m_name{name}
  {}

  virtual ~memory() = default;

  virtual void* allocate(std::size_t n) = 0;
  virtual void deallocate(void* ptr) = 0;

  virtual camp::resources::Platform get_platform() = 0;

  std::size_t get_current_size() const noexcept;
  virtual std::size_t get_actual_size() const noexcept;
  std::size_t get_highwatermark() const noexcept;

  const std::string& get_name() const noexcept
  {
    return m_name;
  }

  int get_id() const noexcept
  {
    return m_id;
  }

  protected:
  template<typename Memory>
  inline void* track_allocation(Memory* self, void* ptr, std::size_t n)
  {
    current_size_ += n;
    actual_size_ += n;
    highwatermark_ = (current_size_ > highwatermark_) 
      ? current_size_ : highwatermark_;

    auto& map = umpire::detail::registry::get()->get_allocation_map();
    map.insert(ptr, umpire::allocation_record{ptr, n, self});
    return ptr;
  }

  inline allocation_record untrack_allocation(void* ptr)
  {
    auto& map = umpire::detail::registry::get()->get_allocation_map();
    auto record = map.remove(ptr);
    current_size_ -= record.size;
    actual_size_ -= record.size;
    return record;
  }

  std::size_t current_size_{0};
  std::size_t actual_size_{0};
  std::size_t highwatermark_{0};
   
  zero_byte_pool_t* zero_pool_;

  private:

  const int m_id;
  const std::string m_name;
};

}
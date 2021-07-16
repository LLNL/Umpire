#pragma once

#include "umpire/detail/allocation_map.hpp"

#include <unordered_map>
#include <list>
#include <mutex>

namespace umpire {

class memory;

namespace detail {

class registry {
public:
  using allocator_list = std::list<std::unique_ptr<umpire::memory>>;
  using allocator_map = std::unordered_map<std::string, umpire::memory*>;
  using allocator_id_map = std::unordered_map<int, umpire::memory*>;

  static registry* get() {
    static registry self;
    return &self;
  }

  int get_id() {
    const std::lock_guard<std::mutex> lock{mtx_};
    return next_allocator_id_++;
  }

  template<typename odr=void>
  allocator_list&
  get_allocator_list()
  {
    return allocator_list_;
  }

  template<typename odr=void>
  allocator_map&
  get_allocator_name_map()
  {
    return allocators_by_name_;
  }

  template<typename odr=void>
  allocator_id_map&
  get_allocator_id_map()
  {
    return allocators_by_id_;
  }

  template<typename odr=void>
  allocation_map&
  get_allocation_map()
  {
    return allocations_;
  }

  ~registry() {
    for (auto&& allocator : allocator_list_) {
      allocator.reset();
    }

  }
  registry (const registry&) = delete;
  registry& operator= (const registry&) = delete;

  private:
  registry() = default;

  allocator_list allocator_list_;
  allocator_map allocators_by_name_;
  allocator_id_map allocators_by_id_;
  allocation_map allocations_;

  // strategy::fixed_pool

  std::mutex mtx_;
  int next_allocator_id_{0};
};


} // end of namespace detail
} // end of namespace umpire
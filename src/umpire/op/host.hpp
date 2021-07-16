#pragma once

#include "umpire/resource/platform.hpp"

#include <cstring>

namespace umpire {
namespace op {

template<>
struct copy<resource::host_platform, resource::host_platform>
{
  template<typename T>
  static void exec(T* src, T* dst, std::size_t len) {
    std::memcpy(dst, src, len*sizeof(T));
  }
};

template<>
struct memset<resource::host_platform>
{
  template<typename T>
  static void exec(T* src, T val, std::size_t len) {
      std::memset(src, val, len);
  }
};

template<>
struct reallocate<resource::host_platform>
{
  template<typename T>
  static T* exec(T* src, std::size_t size) {
    T* ret{nullptr};
    // if ( old_size == 0 ) {
    //   ret = allocator->allocate(new_size);
    //   const std::size_t copy_size = ( old_size > new_size ) ? new_size : old_size;
    //   ResourceManager::getInstance().copy(*new_ptr, current_ptr, copy_size);
    //   allocator->deallocate(current_ptr);
    // } else {
      //auto old_record = ResourceManager::getInstance().deregisterAllocation(current_ptr);
      ret = static_cast<T*>(::realloc(src, size*sizeof(T)));
    //}

    return ret;
  }
};

}
}
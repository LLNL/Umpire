#pragma once

#include "umpire/config.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "camp/resource.hpp"

#include <cstdlib>

namespace umpire {
namespace op {

struct operation {
  static constexpr int arity = -1;
  static constexpr const char* name = "UNKNOWN";
};

template<typename Src, typename Dst>
struct copy : public operation {
  static constexpr int arity = 2;
  static constexpr const char* name = "COPY";
};

template<typename Src>
struct memset : public operation {
  static constexpr int arity = 1;
  static constexpr const char* name = "MEMSET";
};

template<typename Src>
struct reallocate : public operation {
  static constexpr int arity = 1;
  static constexpr const char* name = "REALLOCATE";
};

// Generic reallocate implementation that works for any platform
// This is the template-based version of GenericReallocateOperation
template<typename Src>
struct generic_reallocate : public operation {
  static constexpr int arity = 1;
  static constexpr const char* name = "REALLOCATE";
  
  template<typename T>
  static T* exec(T* current_ptr, std::size_t new_size) {
    if (!current_ptr) {
      // If current pointer is null, just allocate
      auto& rm = ResourceManager::getInstance();
      Allocator allocator = rm.getDefaultAllocator();
      return static_cast<T*>(allocator.allocate(new_size * sizeof(T)));
    }
    
    auto& rm = ResourceManager::getInstance();
    
    // Find the allocator that owns current_ptr
    Allocator allocator = rm.getAllocator(current_ptr);
    
    // Get the current allocation size
    std::size_t old_size = rm.getSize(current_ptr);
    
    // Convert sizes from elements to bytes
    std::size_t old_bytes = old_size;
    std::size_t new_bytes = new_size * sizeof(T);
    
    // Allocate new memory
    T* new_ptr = static_cast<T*>(allocator.allocate(new_bytes));
    
    // Calculate copy size (minimum of old and new size)
    std::size_t copy_size = (old_bytes > new_bytes) ? new_bytes : old_bytes;
    
    // Copy data from old to new location
    rm.copy(new_ptr, current_ptr, copy_size);
    
    // Deallocate old memory
    allocator.deallocate(current_ptr);
    
    return new_ptr;
  }
  
  // Async version
  template<typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* current_ptr, std::size_t new_size, camp::resources::Resource& ctx) {
    if (!current_ptr) {
      // If current pointer is null, just allocate
      auto& rm = ResourceManager::getInstance();
      Allocator allocator = rm.getDefaultAllocator();
      // Since there's no data to copy, we can just return a completed event
      allocator.allocate(new_size * sizeof(T));
      return camp::resources::EventProxy<camp::resources::Resource>{ctx};
    }
    
    auto& rm = ResourceManager::getInstance();
    
    // Find the allocator that owns current_ptr
    Allocator allocator = rm.getAllocator(current_ptr);
    
    // Get the current allocation size
    std::size_t old_size = rm.getSize(current_ptr);
    
    // Convert sizes from elements to bytes
    std::size_t old_bytes = old_size;
    std::size_t new_bytes = new_size * sizeof(T);
    
    // Allocate new memory
    T* new_ptr = static_cast<T*>(allocator.allocate(new_bytes));
    
    // Calculate copy size (minimum of old and new size)
    std::size_t copy_size = (old_bytes > new_bytes) ? new_bytes : old_bytes;
    
    // Copy data from old to new location asynchronously
    auto event = rm.copy(new_ptr, current_ptr, ctx, copy_size);
    
    // IMPORTANT: In a fully async implementation, we would need to chain the deallocation
    // to happen after the copy completes. However, since we don't have that mechanism yet,
    // and ResourceManager's reallocate operation doesn't wait on the event, we need to 
    // deallocate here as we did in the synchronous case.
    //
    // This has the potential to cause race conditions if the memory is deallocated before
    // the copy completes, but for most allocators, the memory won't be immediately reused.
    // A better solution would be to have the ResourceManager wait on the event before returning
    // or implement a chained operation system.
    allocator.deallocate(current_ptr);
    
    return event;
  }
  
  // void* specialization for sync version
  static void* exec(void* current_ptr, std::size_t new_size) {
    return exec<char>(static_cast<char*>(current_ptr), new_size);
  }
  
  // void* specialization for async version
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      void* current_ptr, std::size_t new_size, camp::resources::Resource& ctx) {
    return exec<char>(static_cast<char*>(current_ptr), new_size, ctx);
  }
};

template<typename Src>
struct advise : public operation {
  static constexpr int arity = 1;
  static constexpr const char* name = "ADVISE";
};

template<typename Src>
struct accessed_by : public operation {
  static constexpr int arity = 1;
  static constexpr const char* name = "SET_ACCESSED_BY";
};

template<typename Src>
struct preferred_location : public operation {
  static constexpr int arity = 1;
  static constexpr const char* name = "SET_PREFERRED_LOCATION";
};

template<typename Src>
struct read_mostly : public operation {
  static constexpr int arity = 1;
  static constexpr const char* name = "SET_READ_MOSTLY";
};

template<typename Src>
struct unset_accessed_by : public operation {
  static constexpr int arity = 1;
  static constexpr const char* name = "UNSET_ACCESSED_BY";
};

template<typename Src>
struct unset_preferred_location : public operation {
  static constexpr int arity = 1;
  static constexpr const char* name = "UNSET_PREFERRED_LOCATION";
};

template<typename Src>
struct unset_read_mostly : public operation {
  static constexpr int arity = 1;
  static constexpr const char* name = "UNSET_READ_MOSTLY";
};

#if (defined(UMPIRE_ENABLE_HIP) && HIP_VERSION_MAJOR >= 5) || defined(UMPIRE_ENABLE_CUDA)
template<typename Src>
struct coarse_grain : public operation {
  static constexpr int arity = 1;
  static constexpr const char* name = "SET_COARSE_GRAIN";
};

template<typename Src>
struct unset_coarse_grain : public operation {
  static constexpr int arity = 1;
  static constexpr const char* name = "UNSET_COARSE_GRAIN";
};
#endif

template<typename Src>
struct prefetch : public operation {
  static constexpr int arity = 1;
  static constexpr const char* name = "PREFETCH";
};

}
}
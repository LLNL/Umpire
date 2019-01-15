// This file setups up various definitions for unit testing.
//
//   - Defines AllocatorType
//   - Defines STLAllocator according to the std::allocator concept.
//   - Redefines the new operator.

#ifndef _ALLOCATORTEST_HPP
#define _ALLOCATORTEST_HPP

#include <limits>

#include "StdAllocator.hpp"
#include "DynamicSizePool.hpp"

// If USE_CUDA is defined, test with a managed allocation
#if defined(USE_CUDA)
#include "cuda_runtime.h"

struct CUDAAllocator
{
  static inline void *allocate(std::size_t size) {
    void *ptr;
    cudaMalloc(&ptr, size);
    return ptr;
  }
  static inline void deallocate(void *ptr) { cudaFree(ptr); }
};

#if defined(USE_UVM)
struct UVMAllocator
{
  static inline void *allocate(std::size_t size) {
    void *ptr;
    cudaMallocManaged(&ptr, size);
    return ptr;
  }
  static inline void deallocate(void *ptr) { cudaFree(ptr); }
};

typedef UVMAllocator AllocatorType;
#elif defined(USE_CUDA_HOSTALLOC)
struct CUDAHostAllocator
{
  static inline void *allocate(std::size_t size) {
    void *ptr;
    cudaHostAlloc(&ptr, size);
    return ptr;
  }
  static inline void deallocate(void *ptr) { cudaFree(ptr); }
};
typedef CUDAHostAllocator AllocatorType;
#else
typedef CUDAAllocator AllocatorType;
#endif

// If USE_OMP is defined, test with omp_target_alloc
#elif defined(USE_OMP)
#include <omp.h>

struct OMPAllocator
{
  static inline void *allocate(std::size_t size) {
    void *ptr;
    omp_target_alloc(&ptr, omp_get_default_device());
    return ptr;
  }
  static inline void deallocate(void *ptr) { omp_target_free(ptr); }
};

typedef OMPAllocator AllocatorType;

#else
// Else, use the default StdAllocator from StdAllocator.hpp
typedef StdAllocator AllocatorType;

#endif

// Declare an STL allocator
template <class T>
struct STLAllocator {
  typedef T value_type;
  typedef std::size_t size_type;

  typedef DynamicSizePool<AllocatorType> PoolType;

  PoolType &m;

  STLAllocator() : m(PoolType::getInstance()) { }
  STLAllocator(const STLAllocator &other) { }

  T* allocate(std::size_t n) { return static_cast<T*>( m.allocate( n * sizeof(T) ) ); }
  void deallocate(T* p, std::size_t n) { m.deallocate(p); }

  size_type max_size() const { return std::numeric_limits<unsigned int>::max(); }
};

template <class T, class U>
bool operator==(const STLAllocator<T>&, const STLAllocator<U>&) { return true; }
template <class T, class U>
bool operator!=(const STLAllocator<T>&, const STLAllocator<U>&) { return false; }


#if defined(REDEFINE_NEW)
// Redefine "::new(std::size_t)"

void *operator new (std::size_t size) throw (std::bad_alloc)
{
  return DynamicSizePool<AllocatorType>::getInstance().allocate(size);
}

void *operator new[] (std::size_t size) throw (std::bad_alloc)
{
  return DynamicSizePool<AllocatorType>::getInstance().allocate(size);
}

void operator delete (void *ptr) throw()
{
  return DynamicSizePool<AllocatorType>::getInstance().deallocate(ptr);
}

void operator delete[] (void *ptr) throw()
{
  return DynamicSizePool<AllocatorType>::getInstance().deallocate(ptr);
}

#endif

#endif

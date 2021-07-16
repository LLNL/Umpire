//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "camp/resource/platform.hpp"

namespace umpire {

struct undefined_platform_tag{};
struct host_platform_tag{};
struct cuda_platform_tag{};

struct allocation_strategy
{
  using platform = undefined_platform_tag;

  virtual void* allocate(std::size_t n) = 0;
  virtual void deallocate(void* ptr, std::size_t n) = 0;

  virtual camp::resources::Platform get_platform()
  {
    return camp::resources::Platform::undefined;
  }
};

struct host_resource : public allocation_strategy
{
  using platform = host_platform_tag;

  static host_resource* get_instance() {
    static host_resource self;
    return &self;
  }

  void* allocate(std::size_t n) override
  {
    return malloc(n);
  }

  void deallocate(void* ptr, std::size_t) override
  {
    free(ptr);
  }

  camp::resources::Platform get_platform() override
  {
    return camp::resources::Platform::host;
  }

  private:
    host_resource() = default;
};

using host = host_resource;

template<typename Memory=allocation_strategy>
struct pool : public allocation_strategy
{
  using platform = typename Memory::platform;

  pool() : memory{Memory::get_instance()} {}
  pool(Memory* m) : memory{m} {}

  void* allocate(std::size_t n) override
  {
    return memory->allocate(n);
  }

  void deallocate(void* ptr, std::size_t) override
  {
    memory->deallocate(ptr, 0);
  }

  camp::resources::Platform get_platform() override
  {
    return memory->get_platform();
  }

  Memory* memory;
};

template<class T, typename Memory=allocation_strategy>
class allocator {
  public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using value_type = T;

    allocator() : m_strategy{new Memory} {}
    allocator(Memory* strategy) : m_strategy{strategy}{}
    allocator(allocator const& o) : m_strategy{o.m_strategy} {}

    allocator& operator=(allocator const&) = default;
    bool operator==(allocator const& o) const{return m_strategy == o.m_strategy;}
    bool operator!=(allocator const& o) const{return m_strategy != o.m_strategy;}

    pointer allocate(size_type n){
      return static_cast<pointer>(m_strategy->allocate(n*sizeof(value_type)));
    }

    void deallocate(pointer ptr, std::size_t n = static_cast<std::size_t>(-1)){
      if (ptr) {
        m_strategy->deallocate(ptr, n);
      }
    }

    camp::resources::Platform get_platform() noexcept
    {
      return m_strategy->get_platform();
    }
  private:
    Memory* m_strategy;
};

using Allocator = allocator<char>;


enum class OperationKind {
  copy,
  memset,
  reallocate
};

struct op_base {};

template<OperationKind op>
struct op {
  static constexpr OperationKind kind = op;
}

template <typename... Platforms>
struct copy_op {};

template<>
struct copy_op<host_platform_tag, host_platform_tag>
{
  template<typename T>
  static void exec(T* src, T* dst, std::size_t len) {
    std::cout << "static host copy";
    std::memcpy(dst, src, len*sizeof(T));
  }
};

template<>
struct copy_op<undefined_platform_tag, undefined_platform_tag>
{
  template<typename T, typename SrcMemory, typename DstMemory>
  static void exec(T* src, T* dst, std::size_t len, SrcMemory* src_mem, DstMemory* dst_mem) {
    if (src_mem->get_platform() == dst_mem->get_platform() && src_mem->get_platform() == camp::resources::Platform::host) {
      std::cout << "dynamic host copy";
      std::memcpy(dst, src, len*sizeof(T));
    } else {
      std::cout << "cannot find copy";
    }
  }
};

template<typename T>
void copy(T* src, T* dst, std::size_t len) {
  // logic ot get allocators here
  allocation_strategy* blah{host::get_instance()};
  copy_op<undefined_platform_tag, undefined_platform_tag>::template exec<T>(
    src, dst, len, blah, blah);
}

template<typename SrcMemory, typename DstMemory, typename T>
void copy(T* src, T* dst, std::size_t len) {
  #ifdef DEBUG
  // dynamic lookup
  #endif
  copy_op<typename SrcMemory::platform, typename DstMemory::platform>::template exec<T>(src, dst, len);
}

}


//#include "umpire/allocator.hpp"
//#include "umpire/allocation_strategy.hpp"

#include <iostream>
#include <random>

int main() {
  // get generic pointer to a strategy
  umpire::allocation_strategy* strategy{ /* rm.getAllocator() */umpire::host::get_instance()};
  umpire::Allocator allocator(strategy);
  //umpire::allocator<int> allocator{strategy};

  umpire::allocator<int, umpire::host> host_allocator{};
  umpire::allocator<int> generic_host_allocator{ umpire::host::get_instance()};

  // specific typedef for the pool, with a host resource
  using pool_allocator = umpire::pool<umpire::host>;
  using persistent_allocator = umpire::host;;

  umpire::allocator<int, p> pool{"MY_POOL"};

  // generic pool
  umpire::allocation_strategy* pool_generic{new umpire::pool<>{strategy}};
  umpire::Allocator test{pool_generic};

  // pool with no specific allocator
  umpire::pool<> my_pool{strategy};
  umpire::allocator<char, umpire::pool<>> test_pool{&my_pool};

  int* data = allocator.allocate(100);
  int* pool_data = pool.allocate(100);

  for (int i{0}; i < 100; i++) {
    data[i] = i;
  }

  umpire::copy<p, umpire::host>(data, pool_data, 100);

  for (int i{0}; i < 100; i++) {
    if (pool_data[i] != i) {
      std::cout << "Uh oh" << std::endl;
    }
    pool_data[i] = i*2;
  }

  umpire::copy(pool_data, data, 100);

  for (int i{0}; i < 100; i++) {
    if (data[i] != i*2) {
      std::cout << "Uh oh" << std::endl;
    }
  }
}
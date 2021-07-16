//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "umpire/resource/memory_resource.hpp"

#include <memory>
#include <sstream>

#if !defined(_MSC_VER)
#include <sys/mman.h>
#else
#include <windows.h>
#endif

namespace umpire {
namespace resource {

template<bool Tracking=true>
class null_memory :
  public memory_resource<umpire::resource::undefined_platform>
{
  public:
  static null_memory* get() {
    static null_memory self;
    return &self;
  }

  void* allocate(std::size_t n) override
  {
#if !defined(_MSC_VER)
    void* ptr{mmap(NULL, bytes, PROT_NONE, (MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE), -1, 0)};
#else
    void* ptr{VirtualAlloc(NULL, bytes, MEM_RESERVE, PAGE_NOACCESS)};
#endif
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
    m_size_map.insert(ptr, bytes);

    if constexpr(Tracking) {
      return this->track_allocation(this, ptr, n);
    } else {
      return ptr;
    }
  }

  void deallocate(void* ptr) override
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

    if constexpr(Tracking) {
      this->untrack_allocation(ptr);
    }

    auto iter = m_size_map.find(ptr);
    auto size = iter->second;
    m_size_map.erase(ptr);

#if !defined(_MSC_VER)
    munmap(ptr, *size);
#else
    VirtualFree(ptr, *size, MEM_RELEASE);
#endif
  }

  camp::resources::Platform get_platform() override
  {
    return camp::resources::Platform::undefined;
  }

  ~null_memory() = default;
  null_memory(const null_memory&) = delete;
  null_memory& operator=(const null_memory&) = delete;

  private:
    null_memory() :
      memory_resource<umpire::resource::undefined_platform>{"__null_memory"}
    {}

  util::MemoryMap<size_t> size_by_ptr_;

};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_NullMemoryResource_HPP

//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_SICMAllocator_HPP
#define UMPIRE_SICMAllocator_HPP

#include <cstdlib>
#include <list>
#include <map>
#include <sched.h>
#include <vector>

#include "umpire/util/Macros.hpp"
#include "umpire/util/SICM_device.hpp"

#include <sicm_low.h>

namespace umpire {
namespace alloc {

/*!
 * \brief Uses SICM to allocate and deallocate memory into arenas.
 */
struct SICMAllocator
{
  SICMAllocator(const std::string& name, sicm_device_list* devices)
    : name(name),
      arena(sicm_arena_create(0, static_cast<sicm_arena_flags>(0), devices))
  {
    if (!arena) {
      UMPIRE_ERROR("SICMAllocator could not create an arena with given device list");
    }
  }

  ~SICMAllocator()
  {
    sicm_arena_destroy(arena);
  }

  /*!
   * \brief Allocate bytes of memory using sicm_alloc.
   *
   * \param bytes Number of bytes to allocate.
   * \return Pointer to start of the allocation.
   *
   * \throws umpire::util::Exception if memory cannot be allocated.
   */
  void* allocate(size_t bytes)
  {
    void* ret = sicm_arena_alloc(arena, bytes);

    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);
    if  (ret == nullptr) {
      UMPIRE_ERROR("SICM( bytes = " << bytes << " ) failed");
    } else {
      return ret;
    }
  }

  /*!
   * \brief Deallocate memory using sicm_free.
   *
   * \param ptr Address to deallocate.
   *
   * \throws umpire::util::Exception if memory cannot be free'd.
   */
  void deallocate(void* ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
    sicm_free(ptr);
  }

  const std::string name;
  sicm_arena arena;
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_SICMAllocator_HPP

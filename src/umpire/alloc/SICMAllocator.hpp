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

#include "umpire/util/Macros.hpp"

extern "C"
{
#include <sicm_low.h>
}

namespace umpire {
namespace alloc {

/*!
 * \brief Uses SICM and free to allocate and deallocate CPU memory.
 */
struct SICMAllocator
{
  // SICMAllocator()
  //   : devs(sicm_init()),
  //     sa(sicm_arena_create(0, &devs.devices[0]))
  // {
  //     if (!sa) {
  //       UMPIRE_ERROR("SICMAllocator Could not create arena");
  //     }
  // }

  /*!
   * \brief Allocate bytes of memory using SICM.
   *
   * \param bytes Number of bytes to allocate.
   * \return Pointer to start of the allocation.
   *
   * \throws umpire::util::Exception if memory cannot be allocated.
   */
  void* allocate(size_t bytes)
  {
    // void* ret = sicm_arena_alloc(sa, bytes);
    void* ret = ::sicm_alloc(bytes);
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);
     if  (ret == nullptr) {
      UMPIRE_ERROR("SICM( bytes = " << bytes << " ) failed");
    } else {
      return ret;
    }
  }

  /*!
   * \brief Deallocate memory using free.
   *
   * \param ptr Address to deallocate.
   *
   * \throws umpire::util::Exception if memory cannot be free'd.
   */
  void deallocate(void* ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
    ::sicm_free(ptr);
  }

  // sicm_device_list devs;
  // sicm_arena sa;
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_SICMAllocator_HPP

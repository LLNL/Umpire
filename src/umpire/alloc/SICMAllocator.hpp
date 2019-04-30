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
 * \brief Uses SICM to allocate and deallocate memory into arenas.
 */
struct SICMAllocator
{
  // SICMAllocator(unsigned int device = 0)
  //   : devs(sicm_init()),
  //     sa(nullptr)
  // {
  //   if (device >= devs.count) {
  //     sicm_fini();
  //     UMPIRE_ERROR("SICMAllocator Bad device index: " << device << " / " << devs.count);
  //   }

  //   if (!(sa = sicm_arena_create(0, &devs.devices[device]))) {
  //     sicm_fini();
  //     UMPIRE_ERROR("SICMAllocator Could not create arena on device " << device);
  //   }
  // }

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
    // void* ret = sicm_arena_alloc(sa, bytes);
    void* ret = sicm_alloc(bytes);
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);
     if  (ret == nullptr) {
      sicm_fini();
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

  // sicm_device_list devs;
  // sicm_arena sa;
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_SICMAllocator_HPP

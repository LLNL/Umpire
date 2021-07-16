//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "omp.h"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace alloc {

/*!
 * \brief Uses malloc and free to allocate and deallocate CPU memory.
 */
struct omp_target_allocator
{
  /*!
   * \brief Allocate bytes of memory using malloc.
   *
   * \param bytes Number of bytes to allocate.
   * \return Pointer to start of the allocation.
   *
   * \throws umpire::util::Exception if memory cannot be allocated.
   */
  static void* allocate(std::size_t bytes, int device=0)
  {
    void* ret = omp_target_alloc(bytes, device);
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

    if  (ret == nullptr) {
      UMPIRE_ERROR("omp_target_alloc( bytes = " << bytes << ", device = " << device << " ) failed");
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
  static void deallocate(void* ptr, int device=0)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
    omp_target_free(ptr, device);
  }
};

} // end of namespace alloc
} // end of namespace umpire

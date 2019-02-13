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
#ifndef UMPIRE_PosixMemalignAllocator_HPP
#define UMPIRE_PosixMemalignAllocator_HPP

#include <stdlib.h>

#include "umpire/util/Macros.hpp"
#include "umpire/util/Numa.hpp"

namespace umpire {
namespace alloc {

/*!
 * \brief Uses posix_memalign() and free() to allocate page-aligned memory.
 */
struct PosixMemalignAllocator
{
  /*!
   * \brief Allocate bytes of memory using posix_memalign.
   *
   * \param bytes Number of bytes to allocate. Does not have to be a multiple of the system page size.
   * \return Pointer to start of the allocation.
   *
   * \throws umpire::util::Exception if memory cannot be allocated.
   */
  void* allocate(size_t bytes)
  {
    void* ret = NULL;
    ::posix_memalign(&ret, s_cpu_page_size, bytes);

    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

    if  (ret == nullptr) {
      UMPIRE_ERROR("posix_memalign( bytes = " << bytes << ", pagesize = " << s_cpu_page_size << " ) failed");
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
    ::free(ptr);
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_PosixMemalignAllocator_HPP

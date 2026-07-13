//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_PosixMemalignAllocator_HPP
#define UMPIRE_PosixMemalignAllocator_HPP

#include <stdlib.h>

#include <cerrno>

#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"
#include "umpire/util/numa.hpp"

namespace umpire {
namespace alloc {

/*!
 * \brief Uses posix_memalign() and free() to allocate page-aligned memory.
 */
struct PosixMemalignAllocator {
  /*!
   * \brief Allocate bytes of memory using posix_memalign.
   *
   * \param bytes Number of bytes to allocate. Does not have to be a multiple of
   * the system page size. \return Pointer to start of the allocation.
   *
   * \throws umpire::util::runtime_error if memory cannot be allocated.
   */
  void* allocate(std::size_t bytes)
  {
    void* ret = NULL;
    int err = ::posix_memalign(&ret, get_page_size(), bytes);

    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

    if (ret == nullptr) {
      if (err == ENOMEM) {
        UMPIRE_ERROR(out_of_memory_error,
                     fmt::format("posix_memalign( bytes = {}, pagesize = {} ) failed with error = {}", bytes,
                                 get_page_size(), strerror(err)));
      } else {
        UMPIRE_ERROR(runtime_error, fmt::format("posix_memalign( bytes = {}, pagesize = {} ) failed with error = {}",
                                                bytes, get_page_size(), strerror(err)));
      }
    }

    return ret;
  }

  /*!
   * \brief Deallocate memory using free.
   *
   * \param ptr Address to deallocate.
   *
   * \throws umpire::util::runtime_error if memory cannot be free'd.
   */
  void deallocate(void* ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
    ::free(ptr);
  }

  bool isAccessible(Platform p)
  {
    if (p == Platform::host)
      return true;
    else
      return false;
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_PosixMemalignAllocator_HPP

//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_NoOpAllocator_HPP
#define UMPIRE_NoOpAllocator_HPP

#include <cstdlib>

#include "umpire/util/Macros.hpp"
#include "umpire/util/Platform.hpp"

namespace umpire {
namespace alloc {

/*!
 * \brief Uses malloc and free to allocate and deallocate CPU memory.
 */
struct NoOpAllocator {
  /*!
   * \brief Allocate bytes of memory using malloc.
   *
   * \param bytes Number of bytes to allocate.
   * \return Pointer to start of the allocation.
   *
   * \throws umpire::util::Exception if memory cannot be allocated.
   */
  void* allocate(std::size_t bytes)
  {
    void* ret = ::malloc(bytes);
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

    if (ret == nullptr) {
      UMPIRE_ERROR("malloc( bytes = " << bytes << " ) failed");
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

  bool isAccessible(Platform p)
  {
    if(p == Platform::host)
      return true;
    else  
      return false;
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_NoOpAllocator_HPP

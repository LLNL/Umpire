//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_OpenMPTargetAllocator_HPP
#define UMPIRE_OpenMPTargetAllocator_HPP

#include "omp.h"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace alloc {

/*!
 * \brief Uses malloc and free to allocate and deallocate CPU memory.
 */
struct OpenMPTargetAllocator {
  OpenMPTargetAllocator(int _device) : device{_device}
  {
  }
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
    void* ret = omp_target_alloc(bytes, device);
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

    if (ret == nullptr) {
      UMPIRE_ERROR("omp_target_alloc( bytes = " << bytes << ", device = "
                                                << device << " ) failed");
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
    omp_target_free(ptr, device);
  }

  int device;
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_MallocAllocator_HPP

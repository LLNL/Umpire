//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
#ifndef UMPIRE_AmAllocAllocator_HPP
#define UMPIRE_AmAllocAllocator_HPP

#include <hc_am.hpp>

namespace umpire {
namespace alloc {

/*!
 * \brief Uses hcAlloc and hcAlloc to allocate and deallocate memory on
 *        AMD GPUs that support ROCm.
 */
struct AmAllocAllocator {
  /*!
   * \brief Allocate bytes of memory using hcAlloc
   *
   * \param bytes Number of bytes to allocate.
   * \return Pointer to start of the allocation.
   *
   * \throws umpire::util::Exception if memory cannot be allocated.
   */
  void* allocate(size_t bytes)
  {
    /* Default accelerator */
    hc::accelerator acc;
    void* ret = hc::am_alloc(bytes, acc, 0);

    UMPIRE_LOG(Debug, "(size=" << bytes << ") returning " << ret);

    if  (ret == nullptr) {
      UMPIRE_ERROR("hc::am_alloc(bytes = " << bytes << ") failed");
    } else {
      return ret;
    }
  }

  /*!
   * \brief Deallocate memory using hcFree.
   *
   * \param ptr Address to deallocate.
   *
   * \throws umpire::util::Exception if memory cannot be free'd.
   */
  void deallocate(void* ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
    auto status = hc::am_free(ptr);
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_AmAllocAllocator_HPP

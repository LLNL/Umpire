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
#ifndef UMPIRE_AmPinnedAllocator_HPP
#define UMPIRE_AmPinnedAllocator_HPP

#include <hc_am.hpp>

namespace umpire {
namespace alloc {

/*!
 * \brief Uses hcAlloc and hcAlloc to allocate and deallocate memory on
 *        AMD GPUs that support ROCm.
 */
struct AmPinnedAllocator {
  /*!
   * \brief Allocate bytes of pinned memory using am_alloc
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
    void* ret = hc::am_alloc(bytes, acc, hc::amHostPinned);

    UMPIRE_LOG(Debug, "(size=" << bytes << ") returning " << ret);

    if  (ret == nullptr) {
      UMPIRE_ERROR("hc::am_alloc(bytes = " << bytes
          << ", acc=" << 0
          << ", hc::amHostPinned) failed");
    } else {
      return ret;
    }
  }

  /*!
   * \brief Deallocate memory using am_free.
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

#endif // UMPIRE_AmPinnedAllocator_HPP

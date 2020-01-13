//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
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
  void* allocate(std::size_t bytes)
  {
    /* Default accelerator */
    hc::accelerator acc;
    void* ret = hc::am_alloc(bytes, acc, amHostPinned);

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

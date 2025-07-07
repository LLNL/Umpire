////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_device_allocator_helper_HPP
#define UMPIRE_device_allocator_helper_HPP

#include <string.h>

#include "umpire/DeviceAllocator.hpp"
#include "umpire/util/error.hpp"

namespace umpire {

// Namespace needed to make the macro_tracking var within the
// translation unit scope
namespace {
static int macro_tracking{0};
inline int eliminate_warning_for_macro_tracking()
{
  return macro_tracking;
}
} // namespace

/*
 * Const variable for the limit of unique DeviceAllocator
 * objects available at once.
 */
constexpr int UMPIRE_TOTAL_DEV_ALLOCS{64};

/*
 * Global arrays for both host and device which hold the
 * DeviceAllocator objects created.
 */
extern DeviceAllocator* UMPIRE_DEV_ALLOCS_h;
__device__ extern DeviceAllocator* UMPIRE_DEV_ALLOCS;

/*
 * Get the DeviceAllocator object specified by either the given
 * name or id.
 */
__host__ __device__ DeviceAllocator get_device_allocator(const char* name);
__host__ __device__ DeviceAllocator get_device_allocator(int id);

/*
 * Check if the DeviceAllocator object specified by the
 * given name or id currently exists.
 */
__host__ __device__ bool is_device_allocator(int id);
__host__ __device__ bool is_device_allocator(const char* name);

/*!
 * \brief Construct a new DeviceAllocator. Calls the private Device
 * Allocator constructor that records the associated id.
 *
 * \param allocator Allocator to build the DeviceAllocator from.
 * \param size Total size of the DeviceAllocator.
 * \param name of the DeviceAllocator
 */
__host__ DeviceAllocator make_device_allocator(Allocator allocator, size_t size, const std::string& name);

/*
 * Destroy any DeviceAllocator objects currently in existence.
 * Deallocate any memory belonging to object about to be destroyed.
 */
__host__ void destroy_device_allocator();

/*
 * This macro ensures that the host and device global arrays
 * that keep track of the DeviceAllocator objects created are
 * synced up and pointing to each other.
 */
#if defined(UMPIRE_ENABLE_CUDA)
#define UMPIRE_SET_UP_DEVICE_ALLOCATORS()                                                               \
  {                                                                                                     \
    if (umpire::macro_tracking == 0) {                                                                  \
      UMPIRE_LOG(Debug, "Calling cudaMemcpyToSymbol DeviceAllocator macro.");                           \
      cudaError_t err = cudaMemcpyToSymbol(umpire::UMPIRE_DEV_ALLOCS, &umpire::UMPIRE_DEV_ALLOCS_h,     \
                                           sizeof(umpire::DeviceAllocator*));                           \
      if (err != cudaSuccess) {                                                                         \
        UMPIRE_ERROR(umpire::runtime_error,                                                             \
                     fmt::format("cudaMemcpyToSymbol failed with error: {}", cudaGetErrorString(err))); \
      }                                                                                                 \
    }                                                                                                   \
    umpire::macro_tracking = 1;                                                                         \
  }
#elif defined(UMPIRE_ENABLE_HIP)
#define UMPIRE_SET_UP_DEVICE_ALLOCATORS()                                                             \
  {                                                                                                   \
    if (umpire::macro_tracking == 0) {                                                                \
      UMPIRE_LOG(Debug, "Calling hipMemcpyToSymbol DeviceAllocator macro.");                          \
      hipError_t err = hipMemcpyToSymbol(umpire::UMPIRE_DEV_ALLOCS, &umpire::UMPIRE_DEV_ALLOCS_h,     \
                                         sizeof(umpire::DeviceAllocator*));                           \
      if (err != hipSuccess) {                                                                        \
        UMPIRE_ERROR(umpire::runtime_error,                                                           \
                     fmt::format("hipMemcpyToSymbol failed with error: {}", hipGetErrorString(err))); \
      }                                                                                               \
    }                                                                                                 \
    umpire::macro_tracking = 1;                                                                       \
  }
#else
#define UMPIRE_SET_UP_DEVICE_ALLOCATORS()                                              \
  {                                                                                    \
    UMPIRE_LOG(Warning, "Neither HIP nor CUDA enabled. Macro is not doing anything."); \
  }
#endif

} // end of namespace umpire

#endif // UMPIRE_device_allocator_helper_HPP

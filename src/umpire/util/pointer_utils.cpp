//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/util/pointer_utils.hpp"

#include <string>

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

#if defined(UMPIRE_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif

#if defined(UMPIRE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace umpire {
namespace util {

strategy::AllocationStrategy* inferAllocatorFromPointer(void* ptr, ResourceManager& rm)
{
  if (!ptr) {
    return nullptr;
  }

#if defined(UMPIRE_ENABLE_CUDA)
  {
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);

    if (err == cudaSuccess) {
      // Determine allocator based on memory type
#if CUDART_VERSION >= 10000
      switch (attrs.type) {
        case cudaMemoryTypeUnregistered:
          // Host memory not registered with CUDA - use HOST allocator
          try {
            return rm.getAllocator("HOST").getAllocationStrategy();
          } catch (...) {
            return nullptr;
          }

        case cudaMemoryTypeHost:
          // Pinned host memory
          try {
            return rm.getAllocator("PINNED").getAllocationStrategy();
          } catch (...) {
            // Fallback to HOST if PINNED not available
            try {
              return rm.getAllocator("HOST").getAllocationStrategy();
            } catch (...) {
              return nullptr;
            }
          }

        case cudaMemoryTypeDevice:
          // Device memory
          try {
            if (attrs.device == 0) {
              return rm.getAllocator("DEVICE").getAllocationStrategy();
            } else {
              // Try device-specific allocator
              std::string device_name = "DEVICE::" + std::to_string(attrs.device);
              try {
                return rm.getAllocator(device_name).getAllocationStrategy();
              } catch (...) {
                // Fallback to default DEVICE
                return rm.getAllocator("DEVICE").getAllocationStrategy();
              }
            }
          } catch (...) {
            return nullptr;
          }

        case cudaMemoryTypeManaged:
          // Unified memory
          try {
            return rm.getAllocator("UM").getAllocationStrategy();
          } catch (...) {
            return nullptr;
          }

        default:
          return nullptr;
      }
#else
      // Pre-CUDA 10.0: use older API
      if (attrs.memoryType == cudaMemoryTypeDevice) {
        try {
          return rm.getAllocator("DEVICE").getAllocationStrategy();
        } catch (...) {
          return nullptr;
        }
      } else if (attrs.memoryType == cudaMemoryTypeHost) {
        try {
          return rm.getAllocator("HOST").getAllocationStrategy();
        } catch (...) {
          return nullptr;
        }
      }
#endif
    } else {
      // cudaPointerGetAttributes failed - likely host memory
      cudaGetLastError();  // Clear error
      try {
        return rm.getAllocator("HOST").getAllocationStrategy();
      } catch (...) {
        return nullptr;
      }
    }
  }
#elif defined(UMPIRE_ENABLE_HIP)
  {
    hipPointerAttribute_t attrs;
    hipError_t err = hipPointerGetAttributes(&attrs, ptr);

    if (err == hipSuccess) {
      switch (attrs.type) {
        case hipMemoryTypeHost:
          // Pinned host memory or regular host
          if (attrs.isManaged) {
            try {
              return rm.getAllocator("UM").getAllocationStrategy();
            } catch (...) {
              return nullptr;
            }
          } else {
            try {
              return rm.getAllocator("PINNED").getAllocationStrategy();
            } catch (...) {
              // Fallback to HOST
              try {
                return rm.getAllocator("HOST").getAllocationStrategy();
              } catch (...) {
                return nullptr;
              }
            }
          }

        case hipMemoryTypeDevice:
          // Device memory
          try {
            if (attrs.device == 0) {
              return rm.getAllocator("DEVICE").getAllocationStrategy();
            } else {
              // Try device-specific allocator
              std::string device_name = "DEVICE::" + std::to_string(attrs.device);
              try {
                return rm.getAllocator(device_name).getAllocationStrategy();
              } catch (...) {
                // Fallback to default DEVICE
                return rm.getAllocator("DEVICE").getAllocationStrategy();
              }
            }
          } catch (...) {
            return nullptr;
          }

        case hipMemoryTypeUnified:
          // Unified memory
          try {
            return rm.getAllocator("UM").getAllocationStrategy();
          } catch (...) {
            return nullptr;
          }

        default:
          return nullptr;
      }
    } else {
      // hipPointerGetAttributes failed - likely host memory
      hipGetLastError();  // Clear error
      try {
        return rm.getAllocator("HOST").getAllocationStrategy();
      } catch (...) {
        return nullptr;
      }
    }
  }
#else
  // No GPU support - assume HOST
  try {
    return rm.getAllocator("HOST").getAllocationStrategy();
  } catch (...) {
    return nullptr;
  }
#endif

  return nullptr;
}

} // end of namespace util
} // end of namespace umpire

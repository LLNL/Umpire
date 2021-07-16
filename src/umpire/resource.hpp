#pragma once

#include "umpire/config.hpp"

#include "umpire/resource/platform.hpp"
#include "umpire/resource/resource_type.hpp"

#include "umpire/resource/host_memory.hpp"
#include "umpire/resource/null_memory.hpp"


#if defined(UMPIRE_ENABLE_CUDA)
#include "umpire/resource/cuda_device_memory.hpp"
#include "umpire/resource/cuda_managed_memory.hpp"
#include "umpire/resource/cuda_pinned_memory.hpp"
#if defined(UMPIRE_ENABLE_CONST)
#include "umpire/resource/cuda_const_memory.hpp"
#endif

namespace umpire {
namespace resource {
  using device_memory = cuda_device_memory;
  using managed_memory = cuda_managed_memory;
  using pinned_memory = cuda_pinned_memory;
#if defined(UMPIRE_ENABLE_CONST)
  using const_memory = cuda_const_memory;
#endif
}
}
#endif

#if defined(UMPIRE_ENABLE_HIP)
#include "umpire/resource/hip_device_memory.hpp"
#include "umpire/resource/hip_pinned_memory.hpp"
#if defined(UMPIRE_ENABLE_CONST)
#include "umpire/resource/hip_const_memory.hpp"
#endif

namespace umpire {
namespace resource {
  using device_memory = hip_device_memory;
  using managed_memory = hip_device_memory;
  using pinned_memory = hip_pinned_memory;
#if defined(UMPIRE_ENABLE_CONST)
  using const_memory = hip_const_memory;
#endif
}
}
#endif

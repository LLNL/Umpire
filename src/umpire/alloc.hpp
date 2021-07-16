#pragma once

#include "umpire/config.hpp"

#include "umpire/alloc/malloc_allocator.hpp"
// #include "umpire/alloc/posix_memalign_allocator.hpp"

#if defined(UMPIRE_ENABLE_CUDA)
#include "umpire/alloc/cuda_malloc_allocator.hpp"
#include "umpire/alloc/cuda_malloc_managed_allocator.hpp"
#include "umpire/alloc/cuda_pinned_allocator.hpp"
#endif

#if defined(UMPIRE_ENABLE_HIP)
#include "umpire/alloc/hip_malloc_allocator.hpp"
#include "umpire/alloc/hip_pinned_allocator.hpp"
#endif

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
#include "umpire/alloc/omp_target_allocator.hpp"
#endif
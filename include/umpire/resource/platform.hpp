#pragma once

#include "camp/resource/platform.hpp"
#include "umpire/config.hpp"

namespace umpire {
namespace resource {

//! Runtime platform enum shared with CAMP resources.
using Platform = camp::resources::Platform;

/*!
 * \brief Map a compile-time API v2 platform tag to its runtime enum value.
 *
 * \tparam Platform Compile-time platform tag.
 */
template <typename Platform>
struct platform_for {};

//! Sentinel tag for resources without a concrete execution backend.
struct undefined_platform {};
//! Compile-time tag for host resources.
struct host_platform {};
#if defined(UMPIRE_ENABLE_CUDA)
//! Compile-time tag for CUDA device resources.
struct cuda_platform {};
#endif
#if defined(UMPIRE_ENABLE_HIP)
//! Compile-time tag for HIP device resources.
struct hip_platform {};
#endif
#if defined(UMPIRE_ENABLE_SYCL)
//! Compile-time tag for SYCL device resources.
struct sycl_platform {};
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
//! Compile-time tag for OpenMP target resources.
struct omp_target_platform {};
#endif

template <>
struct platform_for<undefined_platform> {
  static constexpr camp::resources::Platform value = camp::resources::Platform::undefined;
};

template <>
struct platform_for<host_platform> {
  static constexpr camp::resources::Platform value = camp::resources::Platform::host;
};

#if defined(UMPIRE_ENABLE_CUDA)
template <>
struct platform_for<cuda_platform> {
  static constexpr camp::resources::Platform value = camp::resources::Platform::cuda;
};
#endif
#if defined(UMPIRE_ENABLE_HIP)
template <>
struct platform_for<hip_platform> {
  static constexpr camp::resources::Platform value = camp::resources::Platform::hip;
};
#endif
#if defined(UMPIRE_ENABLE_SYCL)
template <>
struct platform_for<sycl_platform> {
  static constexpr camp::resources::Platform value = camp::resources::Platform::sycl;
};
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
template <>
struct platform_for<omp_target_platform> {
  static constexpr camp::resources::Platform value = camp::resources::Platform::omp_target;
};
#endif

} // namespace resource

//! Convenience alias for `resource::host_platform`.
using host = resource::host_platform;
#if defined(UMPIRE_ENABLE_CUDA)
//! Convenience alias for `resource::cuda_platform`.
using cuda = resource::cuda_platform;
#endif
#if defined(UMPIRE_ENABLE_HIP)
//! Convenience alias for `resource::hip_platform`.
using hip = resource::hip_platform;
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
//! Convenience alias for `resource::omp_target_platform`.
using omp_target = resource::omp_target_platform;
#endif

} // namespace umpire

#pragma once

#include "umpire/config.hpp"

#include "camp/resource/platform.hpp"

namespace umpire {
namespace resource {

template<typename Platform>
struct platform_for {};

struct undefined_platform {};
struct host_platform {};
#if defined(UMPIRE_ENABLE_CUDA)
struct cuda_platform {};
#endif
#if defined(UMPIRE_ENABLE_HIP)
struct hip_platform {};
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
struct omp_target_platform {};
#endif

template<>
struct platform_for<undefined_platform> {
  static constexpr camp::resources::Platform value = camp::resources::Platform::undefined;
};

template<>
struct platform_for<host_platform> {
  static constexpr camp::resources::Platform value = camp::resources::Platform::host;
};

#if defined(UMPIRE_ENABLE_CUDA)
template<>
struct platform_for<cuda_platform> {
  static constexpr camp::resources::Platform camp::resources::Platform::cuda;
}
#endif
#if defined(UMPIRE_ENABLE_HIP)
template<>
struct platform_for<hip_platform> {
  static constexpr camp::resources::Platform camp::resources::Platform::hip;
}
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
template<>
struct platform_for<omp_target_platform> {
  static constexpr camp::resources::Platform camp::resources::Platform::omp_target;
}
#endif


}
}
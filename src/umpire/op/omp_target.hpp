#pragma once

#include "umpire/resource/platform.hpp"

#include "omp.h"

namespace umpire {
namespace op {

template<>
struct copy<resource::omp_target_platform, resource::omp_target_platform>
{
  template<typename T>
  static void exec(T* src, T* dst, std::size_t len) {

    ::omp_target_memcpy(
      dst, 
      src, 
      sizeof(T)*len), 
      0,
      0,
      omp_get_default_device(),
      omp_get_default_device());
  }
};

template<>
struct copy<resource::omp_target_platform, resource::host_platform>
{
  template<typename T>
  static void exec(T* src, T* dst, std::size_t len) {
    ::omp_target_memcpy(
      dst, 
      src, 
      sizeof(T)*len), 
      0,
      0,
      omp_get_initial_device(),
      omp_get_default_device());
  }
};

template<>
struct copy<resource::host_platform, resource::omp_target_platform>
{
  template<typename T>
  static void exec(T* src, T* dst, std::size_t len) {
    ::omp_target_memcpy(
      dst, 
      src, 
      sizeof(T)*len), 
      0,
      0,
      omp_get_default_device(),
      omp_get_initial_device());
  }
};

template<>
struct memset<resource::omp_target_platform>
{
  template <typename T>
  static void exec(T *src, T val, std::size_t len)
  {
    int device = omp_get_default_device;

#pragma omp target is_device_ptr(data_ptr) device(device)
#pragma omp teams distribute parallel for schedule(static, 1)
    for (std::size_t i = 0; i < length; ++i) {
      src[i] = val;
    }
  }
};

}

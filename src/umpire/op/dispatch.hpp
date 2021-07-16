#pragma once

#include "umpire/config.hpp"

#include "umpire/resource/platform.hpp"
#include "umpire/detail/registry.hpp"

#include "camp/resource/platform.hpp"

namespace umpire {
namespace op {


template <int args, template<typename... T> class Op> struct op_caller{};

template<template<typename T> class Op>
struct op_caller<1, Op > {
  template<typename T, typename... Args>
  inline static void exec(T* src, Args... args) {
    auto p = camp::resources::Platform::host;
    // get src and dest platform
    if (p == camp::resources::Platform::host) {
      Op<resource::host_platform>::exec(src, args...);
    } 
#if defined(UMPIRE_ENABLE_CUDA)
    else if (p1 == p2 && (p1 == Platform::device)) {
      Op<host_platform, host_platform>::exec(src, dst, args...);
    }
#endif
  }
};

template<class... Ts>
struct count {
    static constexpr std::size_t value = sizeof...(Ts);
};


template<template<typename... Ts> class Op>
struct op_caller<2, Op> {
  // try calling with Op::arity
  template<typename T, typename... Args>
  inline static void exec(T* src, T* dst, Args... args) {
    auto& allocation_map = detail::registry::get()->get_allocation_map();
    auto src_record = allocation_map.find(src);
    auto dst_record = allocation_map.find(dst);

    auto p1 = src_record->strategy->get_platform();;
    auto p2 = dst_record->strategy->get_platform();;

    // get src and dest platform
    if (p1 == p2 && ( p1 == camp::resources::Platform::host)) {
      return Op<resource::host_platform, resource::host_platform>::exec(src, dst, args...);
    } 
#if defined(UMPIRE_ENABLE_CUDA)
    if (p1 == p2 && (p1 == camp::resources::Platform::cuda)) {
      Op<resource::cuda_platform, resource::cuda_platform>::exec(src, dst, args...);
    } else if (p1 == camp::resources::Platform::host && p2 == camp::resources::Platform::cuda) {
      Op<resource::host_platform, resource::cuda_platform>::exec(src, dst, args...);
    } else if (p1 == camp::resources::Platform::cuda && p2 == camp::resources::Platform::host) {
      Op<resource::cuda_platform, resource::host_platform>::exec(src, dst, args...);
    }
#endif
  }
};

}

template<typename Src, typename Dst, typename T>
void copy(T* src, T* dst, std::size_t len) {
  op::copy<typename Src::platform, typename Dst::platform>::exec(src, dst, len);
}

template <typename T>
void copy(T* src, T* dst, std::size_t len) {
    op::op_caller<2, op::copy>::exec(src, dst, len);
}

template<typename Src, typename T>
void memset(T* a, T v, std::size_t len) {
  op::memset<typename Src::platform>::exec(a, v, len);
}

template <typename T>
void copy(T* src, std::size_t len) {
    op::op_caller<1, op::memset>::exec(src, len);
}

}
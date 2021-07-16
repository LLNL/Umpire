#pragma once

#include "umpire/resource/platform.hpp"

namespace umpire {
namespace op {

template<>
struct copy<resource::hip_platform, resource::hip_platform>
{
  template<typename T>
  static void exec(T* src, T* dst, std::size_t len) {
    ::hipMemcpy(dst, src, sizeof(T)*len), hipMemcpyDeviceToDevice);
  }

  template<typename T>
  static camp::resources::Event exec(T* src, T* dst, std::size_t len, camp::resources::Resource& ctx) {
    auto device = ctx.get<camp::resources::Cuda>();
    auto stream = device.get_stream();

    ::hipMemcpyAsync(dst, src, sizeof(T)*len), hipMemcpyDeviceToDevice, stream);
     
    return ctx.getEvent();
  }
};

template<>
struct copy<resource::hip_platform, resource::host_platform>
{
  template<typename T>
  static void exec(T* src, T* dst, std::size_t len) {
    ::hipMemcpy(dst, src, sizeof(T)*len), hipMemcpyDeviceToHost);
  }

  template<typename T>
  static camp::resources::Event exec(T* src, T* dst, std::size_t len, camp::resources::Resource& ctx) {
    auto device = ctx.get<camp::resources::Cuda>();
    auto stream = device.get_stream();

    ::hipMemcpyAsync(dst, src, sizeof(T)*len), hipMemcpyDeviceToHost, stream);
     
    return ctx.getEvent();
  }
};

template<>
struct copy<resource::host_platform, resource::hip_platform>
{
  template<typename T>
  static void exec(T* src, T* dst, std::size_t len) {
    ::hipMemcpy(dst, src, sizeof(T)*len), hipMemcpyHostToDevice);
  }

  template<typename T>
  static camp::resources::Event exec(T* src, T* dst, std::size_t len, camp::resources::Resource& ctx) {
    auto device = ctx.get<camp::resources::Cuda>();
    auto stream = device.get_stream();

    ::hipMemcpyAsync(dst, src, sizeof(T)*len), hipMemcpyHostToDevice, stream);
     
    return ctx.getEvent();
  }
};

template<>
struct memset<resource::hip_platform>
{
  template<typename T>
  static void exec(T* src, T val, std::size_t len) {
    ::hipMemset(src, val, len);
  }

  template<typename T>
  static camp::resources::Event exec(T* src, T val, std::size_t len, camp::resources::Context& ctx) {
    auto device = ctx.get<camp::resources::Cuda>();
    auto stream = device.get_stream();

    ::hipMemsetAsync(src, val, len, stream);

    return ctx.get_event();
  }
};

}
}
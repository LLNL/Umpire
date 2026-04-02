//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/op/copy.hpp"
#include "umpire/op/memset.hpp"
#include "umpire/op/prefetch.hpp"
#include "umpire/op/reallocate.hpp"
#include "umpire/resource/host_memory.hpp"

#include "camp/resource/host.hpp"
#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace {

using host_memory = umpire::resource::host_memory<>;

camp::resources::Resource host_resource()
{
  return camp::resources::Resource{camp::resources::Host{}};
}

host_memory& host()
{
  return host_memory::get();
}

template <typename SrcPlatform, typename DstPlatform, typename = void>
struct has_direct_copy_exec : std::false_type {};

template <typename SrcPlatform, typename DstPlatform>
struct has_direct_copy_exec<
    SrcPlatform,
    DstPlatform,
    std::void_t<decltype(umpire::op::copy<SrcPlatform, DstPlatform>::template exec<int>(
        std::declval<int*>(), std::declval<int*>(), std::declval<std::size_t>()))>> : std::true_type {};

void expect_unsupported_runtime_dispatch(camp::resources::Platform src_platform,
                                         camp::resources::Platform dst_platform)
{
  int src = 7;
  int dst = 0;

  EXPECT_THROW((umpire::op::detail::dispatch<umpire::op::copy>(
                   src_platform, dst_platform, &src, &dst, std::size_t{1})),
               umpire::runtime_error);
}

} // namespace

TEST(ApiV2Operations, DirectHostTemplateCopyUsesElementCounts)
{
  std::array<int, 4> src{{1, 3, 5, 7}};
  std::array<int, 4> dst{{0, 0, 0, 0}};

  umpire::copy<umpire::host_platform, umpire::host_platform>(src.data(), dst.data(), src.size());

  EXPECT_EQ(dst, src);
}

TEST(ApiV2Operations, DirectHostTemplateAsyncOperationsComplete)
{
  auto resource = host_resource();

  std::array<unsigned char, 16> src{};
  std::array<unsigned char, 16> dst{};
  std::fill(src.begin(), src.end(), static_cast<unsigned char>(0x5A));

  auto copy_event =
      umpire::copy<umpire::host_platform, umpire::host_platform>(src.data(), dst.data(), src.size(), resource);
  (void)copy_event;
  resource.get_event().wait();
  EXPECT_EQ(dst, src);

  auto memset_event = umpire::memset<umpire::host_platform>(dst.data(), 0x11, dst.size(), resource);
  (void)memset_event;
  resource.get_event().wait();
  EXPECT_TRUE(std::all_of(dst.begin(), dst.end(), [](unsigned char value) { return value == 0x11; }));

  auto prefetch_event = umpire::prefetch<umpire::host_platform>(dst.data(), 0, dst.size(), resource);
  (void)prefetch_event;
  resource.get_event().wait();
  EXPECT_TRUE(std::all_of(dst.begin(), dst.end(), [](unsigned char value) { return value == 0x11; }));
}

TEST(ApiV2Operations, RuntimeDispatchRejectsPairsWithoutCompiledSupport)
{
  bool exercised = false;

#if !defined(UMPIRE_ENABLE_CUDA)
  exercised = true;
  expect_unsupported_runtime_dispatch(camp::resources::Platform::cuda, camp::resources::Platform::host);
#endif
#if !defined(UMPIRE_ENABLE_HIP)
  exercised = true;
  expect_unsupported_runtime_dispatch(camp::resources::Platform::hip, camp::resources::Platform::host);
#endif
#if !defined(UMPIRE_ENABLE_SYCL)
  exercised = true;
  expect_unsupported_runtime_dispatch(camp::resources::Platform::sycl, camp::resources::Platform::host);
#endif
#if !defined(UMPIRE_ENABLE_OPENMP_TARGET)
  exercised = true;
  expect_unsupported_runtime_dispatch(camp::resources::Platform::omp_target, camp::resources::Platform::host);
#endif

  if (!exercised) {
    GTEST_SKIP() << "All backend dispatch paths are compiled on this build.";
  }
}

TEST(ApiV2Operations, DirectTemplateCopyAvailabilityMatchesCompiledSpecializations)
{
  EXPECT_TRUE((has_direct_copy_exec<umpire::host_platform, umpire::host_platform>::value));

#if defined(UMPIRE_ENABLE_CUDA)
  EXPECT_TRUE((has_direct_copy_exec<umpire::cuda_platform, umpire::cuda_platform>::value));
  EXPECT_TRUE((has_direct_copy_exec<umpire::cuda_platform, umpire::host_platform>::value));
  EXPECT_TRUE((has_direct_copy_exec<umpire::host_platform, umpire::cuda_platform>::value));
#endif
#if defined(UMPIRE_ENABLE_HIP)
  EXPECT_TRUE((has_direct_copy_exec<umpire::hip_platform, umpire::hip_platform>::value));
  EXPECT_TRUE((has_direct_copy_exec<umpire::hip_platform, umpire::host_platform>::value));
  EXPECT_TRUE((has_direct_copy_exec<umpire::host_platform, umpire::hip_platform>::value));
#endif
#if defined(UMPIRE_ENABLE_SYCL)
  EXPECT_TRUE((has_direct_copy_exec<umpire::sycl_platform, umpire::sycl_platform>::value));
  EXPECT_TRUE((has_direct_copy_exec<umpire::sycl_platform, umpire::host_platform>::value));
  EXPECT_TRUE((has_direct_copy_exec<umpire::host_platform, umpire::sycl_platform>::value));
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
  EXPECT_TRUE((has_direct_copy_exec<umpire::omp_target_platform, umpire::omp_target_platform>::value));
  EXPECT_TRUE((has_direct_copy_exec<umpire::omp_target_platform, umpire::host_platform>::value));
  EXPECT_TRUE((has_direct_copy_exec<umpire::host_platform, umpire::omp_target_platform>::value));
#endif

#if defined(UMPIRE_ENABLE_CUDA) && defined(UMPIRE_ENABLE_HIP)
  EXPECT_FALSE((has_direct_copy_exec<umpire::cuda_platform, umpire::hip_platform>::value));
  EXPECT_FALSE((has_direct_copy_exec<umpire::hip_platform, umpire::cuda_platform>::value));
#endif
#if defined(UMPIRE_ENABLE_CUDA) && defined(UMPIRE_ENABLE_SYCL)
  EXPECT_FALSE((has_direct_copy_exec<umpire::cuda_platform, umpire::sycl_platform>::value));
  EXPECT_FALSE((has_direct_copy_exec<umpire::sycl_platform, umpire::cuda_platform>::value));
#endif
#if defined(UMPIRE_ENABLE_CUDA) && defined(UMPIRE_ENABLE_OPENMP_TARGET)
  EXPECT_FALSE((has_direct_copy_exec<umpire::cuda_platform, umpire::omp_target_platform>::value));
  EXPECT_FALSE((has_direct_copy_exec<umpire::omp_target_platform, umpire::cuda_platform>::value));
#endif
#if defined(UMPIRE_ENABLE_HIP) && defined(UMPIRE_ENABLE_SYCL)
  EXPECT_FALSE((has_direct_copy_exec<umpire::hip_platform, umpire::sycl_platform>::value));
  EXPECT_FALSE((has_direct_copy_exec<umpire::sycl_platform, umpire::hip_platform>::value));
#endif
#if defined(UMPIRE_ENABLE_HIP) && defined(UMPIRE_ENABLE_OPENMP_TARGET)
  EXPECT_FALSE((has_direct_copy_exec<umpire::hip_platform, umpire::omp_target_platform>::value));
  EXPECT_FALSE((has_direct_copy_exec<umpire::omp_target_platform, umpire::hip_platform>::value));
#endif
#if defined(UMPIRE_ENABLE_SYCL) && defined(UMPIRE_ENABLE_OPENMP_TARGET)
  EXPECT_FALSE((has_direct_copy_exec<umpire::sycl_platform, umpire::omp_target_platform>::value));
  EXPECT_FALSE((has_direct_copy_exec<umpire::omp_target_platform, umpire::sycl_platform>::value));
#endif
}

TEST(ApiV2Operations, HostResourceBackedBuffersWorkWithOperationTemplates)
{
  auto* src = static_cast<int*>(host().allocate(4 * sizeof(int)));
  auto* dst = static_cast<int*>(host().allocate(4 * sizeof(int)));

  for (int i = 0; i < 4; ++i) {
    src[i] = (i + 1) * 9;
    dst[i] = -1;
  }

  umpire::copy<umpire::host_platform, umpire::host_platform>(src, dst, 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(dst[i], src[i]);
  }

  umpire::memset<umpire::host_platform>(dst, 0, 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(dst[i], 0);
  }

  host().deallocate(src);
  host().deallocate(dst);
}

TEST(ApiV2Operations, V2HostReallocatePreservesTypedContents)
{
  auto* values = static_cast<int*>(host().allocate(4 * sizeof(int)));
  for (int i = 0; i < 4; ++i) {
    values[i] = i + 21;
  }

  values = umpire::reallocate(&values, 8);

  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(values[i], i + 21);
  }

  host().deallocate(values);
}

TEST(ApiV2Operations, V2HostAsyncReallocatePreservesByteContents)
{
  auto resource = host_resource();
  void* ptr = host().allocate(8);
  auto* bytes = static_cast<unsigned char*>(ptr);
  std::fill(bytes, bytes + 8, static_cast<unsigned char>(0xAB));

  auto event = umpire::reallocate(&ptr, 16, resource);
  (void)event;
  resource.get_event().wait();

  bytes = static_cast<unsigned char*>(ptr);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(bytes[i], 0xAB);
  }

  host().deallocate(ptr);
}

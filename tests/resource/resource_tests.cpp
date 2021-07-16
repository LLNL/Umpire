//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/config.hpp"
#include "umpire/resource.hpp"
#include "umpire/detail/tracker.hpp"

#include "umpire/umpire.hpp"

#include "gtest/gtest.h"

static const int blah = []() { umpire::initialize(); return 0; }();

template <typename T>
class resource_test : public ::testing::Test {
};

TYPED_TEST_SUITE_P(resource_test);

TYPED_TEST_P(resource_test, get)
{
  TypeParam* resource = TypeParam::get();
  ASSERT_NE(nullptr, resource);

  auto name = resource->get_name();

  auto resource_by_name = dynamic_cast<TypeParam*>(umpire::get_strategy(name));

  if (resource_by_name)
    std::cout << &(*resource_by_name) << std::endl;
  std::cout << &(*resource) << std::endl;
}

TYPED_TEST_P(resource_test, allocate_deallocate)
{
  TypeParam* resource = TypeParam::get();
  void* ptr{resource->allocate(16)};
  EXPECT_NE(nullptr, ptr);
  resource->deallocate(ptr);
}

TYPED_TEST_P(resource_test, get_platform)
{
  TypeParam* resource = TypeParam::get();

  auto platform = resource->get_platform();
  EXPECT_NE(camp::resources::Platform::undefined, platform);
}

TYPED_TEST_P(resource_test, introspection)
{
  TypeParam* resource = TypeParam::get();

  EXPECT_EQ(resource->get_current_size(), 0);
  EXPECT_EQ(resource->get_actual_size(), 0);
  EXPECT_GE(resource->get_highwatermark(), 0);
}

REGISTER_TYPED_TEST_SUITE_P(
    resource_test,
    get,
    allocate_deallocate,
    get_platform,
    introspection);

using test_types = ::testing::Types<
    umpire::resource::host_memory<>
    , umpire::resource::host_memory<umpire::alloc::malloc_allocator, false>
#if defined(UMPIRE_ENABLE_CUDA)
    , umpire::resource::cuda_device_memory, umpire::resource::cuda_pinned_memory, umpire::resource::cuda_managed_memory
#endif
#if defined(UMPIRE_ENABLE_HIP)
    , umpire::resource::hip_device_memory, umpire::resource::hip_pinned_memory
#endif
>;

INSTANTIATE_TYPED_TEST_SUITE_P(_, resource_test, test_types,);

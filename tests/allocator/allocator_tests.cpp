//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/config.hpp"

#include "umpire/alloc.hpp"
#include "umpire/resource.hpp"
#include "umpire/allocator.hpp"

#include "umpire/umpire.hpp"

#include "gtest/gtest.h"

template <typename T>
class allocator_test : public ::testing::Test {
};

TYPED_TEST_SUITE_P(allocator_test);

TYPED_TEST_P(allocator_test, constructors) {
  using DataType = typename std::tuple_element<0, TypeParam>::type;
  using Memory = typename std::tuple_element<1, TypeParam>::type;

  {
    umpire::allocator<DataType> allocator{Memory::get()};
    EXPECT_FALSE(
      (std::is_same<typename decltype(allocator)::platform, typename Memory::platform>::value) 
    );
    EXPECT_TRUE(
      (std::is_same<typename decltype(allocator)::platform, umpire::resource::undefined_platform>::value) 
    );
    EXPECT_NE(allocator.get_platform(), camp::resources::Platform::undefined);
  }

  {
    umpire::Allocator allocator{Memory::get()};
    EXPECT_FALSE(
      (std::is_same<typename decltype(allocator)::platform, typename Memory::platform>::value) 
    );
    EXPECT_TRUE(
      (std::is_same<typename decltype(allocator)::platform, umpire::resource::undefined_platform>::value) 
    );
    EXPECT_NE(allocator.get_platform(), camp::resources::Platform::undefined);
  }

  {
    umpire::allocator<DataType, Memory> allocator{Memory::get()};
    EXPECT_TRUE( 
      (std::is_same<typename decltype(allocator)::platform, typename Memory::platform>::value) 
    );
    EXPECT_FALSE(
      (std::is_same<typename decltype(allocator)::platform, umpire::resource::undefined_platform>::value) 
    );
    EXPECT_NE(allocator.get_platform(), camp::resources::Platform::undefined);
  }
}

TYPED_TEST_P(allocator_test, allocate_deallocate_strategy) {
  using DataType = typename std::tuple_element<0, TypeParam>::type;
  using Memory = typename std::tuple_element<1, TypeParam>::type;
  umpire::allocator<DataType> allocator{Memory::get()};

  DataType* ptr{allocator.allocate(16)};

  ASSERT_EQ(allocator.get_current_size(), 16);

  EXPECT_NE(nullptr, ptr);

  allocator.deallocate(ptr);
}

TYPED_TEST_P(allocator_test, allocate_deallocate_type) {
  using DataType = typename std::tuple_element<0, TypeParam>::type;
  using Memory = typename std::tuple_element<1, TypeParam>::type;
  umpire::allocator<DataType, Memory> allocator{Memory::get()};

  DataType* ptr{allocator.allocate(16)};

  EXPECT_NE(nullptr, ptr);

  allocator.deallocate(ptr);
}

TYPED_TEST_P(allocator_test, allocate_deallocate_alias) {
  using Memory = typename std::tuple_element<1, TypeParam>::type;
  using DataType = typename umpire::Allocator::pointer;
  umpire::Allocator allocator{Memory::get()};

  DataType ptr{allocator.allocate(16)};

  EXPECT_NE(nullptr, ptr);

  allocator.deallocate(ptr);
}

REGISTER_TYPED_TEST_SUITE_P(
    allocator_test,
    constructors,
    allocate_deallocate_strategy,
    allocate_deallocate_type,
    allocate_deallocate_alias);

using test_types = ::testing::Types<
      std::tuple<double, umpire::resource::host_memory<>>
    , std::tuple<int, umpire::resource::host_memory<>>
    , std::tuple<char, umpire::resource::host_memory<>>
    , std::tuple<float, umpire::resource::host_memory<>>
#if defined(UMPIRE_ENABLE_CUDA)
    , umpire::resource::cuda_device_memory, umpire::resource::cuda_pinned_memory, umpire::resource::cuda_managed_memory
#endif
#if defined(UMPIRE_ENABLE_HIP)
    , umpire::resource::hip_device_memory, umpire::resource::hip_pinned_memory
#endif
>;

INSTANTIATE_TYPED_TEST_SUITE_P(_, allocator_test, test_types,);

class allocator_api_test : public ::testing::TestWithParam<std::string> {
};

TEST_P(allocator_api_test, constructors)
{
  auto allocator = umpire::get_allocator(GetParam());
  (void) allocator;

  SUCCEED();
}

const std::string allocator_strings[] = {
  "HOST"
#if defined(UMPIRE_ENABLE_DEVICE)
    , "DEVICE"
#endif
#if defined(UMPIRE_ENABLE_UM)
    , "UM"
#endif
#if defined(UMPIRE_ENABLE_CONST)
    , "DEVICE_CONST"
#endif
#if defined(UMPIRE_ENABLE_PINNED)
    , "PINNED"
#endif
};

INSTANTIATE_TEST_SUITE_P(
    _,
    allocator_api_test,
    ::testing::ValuesIn(allocator_strings));

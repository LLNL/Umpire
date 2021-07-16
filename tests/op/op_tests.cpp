//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/config.hpp"

#include "umpire/resource.hpp"
#include "umpire/op.hpp"

#include "gtest/gtest.h"

template <typename T>
class op_test : public ::testing::Test {
};

TYPED_TEST_SUITE_P(op_test);

TYPED_TEST_P(op_test, constructors) {
  using DataType = typename std::tuple_element<0, TypeParam>::type;
  using Memory = typename std::tuple_element<1, TypeParam>::type;

  auto mem = Memory::get();
  DataType* src{static_cast<DataType*>(mem->allocate(1024))};
  DataType* dst{static_cast<DataType*>(mem->allocate(1024))};

  umpire::copy<Memory, Memory>(dst, src, 1024);
}

REGISTER_TYPED_TEST_SUITE_P(
  op_test,
  constructors);

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

INSTANTIATE_TYPED_TEST_SUITE_P(_, op_test, test_types,);
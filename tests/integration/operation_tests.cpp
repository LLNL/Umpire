//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"

#include "umpire/config.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/util/Exception.hpp"

#include "umpire/op/MemoryOperationRegistry.hpp"

#if defined(UMPIRE_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif

class OperationTest :
  public ::testing::TestWithParam< ::testing::tuple<std::string, std::string> >
{
  public:
    void SetUp() override {
      auto& rm = umpire::ResourceManager::getInstance();
      source_allocator = new umpire::Allocator(rm.getAllocator(::testing::get<0>(GetParam())));
      dest_allocator = new umpire::Allocator(rm.getAllocator(::testing::get<1>(GetParam())));

      source_array = static_cast<float*>(source_allocator->allocate(m_size*sizeof(float)));
      dest_array = static_cast<float*>(dest_allocator->allocate(m_size*sizeof(float)));

      check_array = static_cast<float*>(
          rm.getAllocator("HOST").allocate(m_size*sizeof(float)));
    }

    void TearDown() override {
      auto& rm = umpire::ResourceManager::getInstance();

      if (source_array)
        source_allocator->deallocate(source_array);

      if (dest_array)
        dest_allocator->deallocate(dest_array);

      if (check_array)
        rm.getAllocator("HOST").deallocate(check_array);

      delete source_allocator;
      delete dest_allocator;
    }

    float* source_array;
    float* dest_array;
    float* check_array;

    const std::size_t m_size = 1024;

    umpire::Allocator* source_allocator;
    umpire::Allocator* dest_allocator;
};

class ZeroCopyTest :
  public OperationTest
{
};

TEST_P(ZeroCopyTest, Zero) {
  auto& rm = umpire::ResourceManager::getInstance();

  void* src = source_allocator->allocate(0);
  void* dst = dest_allocator->allocate(0);

  rm.copy(dst, src);

  source_allocator->deallocate(src);
  dest_allocator->deallocate(dst);
}

class CopyTest :
  public OperationTest
{
};

TEST_P(CopyTest, Copy) {
    auto& rm = umpire::ResourceManager::getInstance();

    for (std::size_t i = 0; i < m_size; i++) {
      source_array[i] = static_cast<float>(i);
    }

    rm.copy(dest_array, source_array);

    rm.copy(check_array, dest_array);

    for (std::size_t i = 0; i < m_size; i++) {
      ASSERT_FLOAT_EQ(source_array[i], check_array[i]);
    }
}

TEST_P(CopyTest, Single)
{
    auto& rm = umpire::ResourceManager::getInstance();

    check_array[10] = 3.14f;

    rm.copy(&dest_array[11], &check_array[10], sizeof(float));
    rm.copy(&check_array[0], &dest_array[11], sizeof(float));

    ASSERT_EQ(check_array[0], check_array[10]);
}

TEST_P(CopyTest, Offset) {
  auto& rm = umpire::ResourceManager::getInstance();

  for (std::size_t i = 0; i < m_size; ++i)
  {
    source_array[i] = static_cast<float>(i);
  }

  rm.copy(dest_array, source_array);

  rm.copy(check_array, dest_array);
  rm.copy(check_array, dest_array + m_size / 2);

  for (std::size_t i = 0; i < m_size / 2; ++i) {
    ASSERT_EQ(i + m_size / 2, check_array[i]);
  }

  for (std::size_t i = m_size / 2; i < m_size; ++i) {
    ASSERT_EQ(i, check_array[i]);
  }
}

TEST_P(CopyTest, InvalidSize)
{
    auto& rm = umpire::ResourceManager::getInstance();

    ASSERT_THROW(
        rm.copy(dest_array, source_array, (m_size+100)*sizeof(float)),
        umpire::util::Exception);

    float* small_dest_array = static_cast<float*>(
        dest_allocator->allocate(10*sizeof(float)));

    ASSERT_THROW(
        rm.copy(small_dest_array, source_array),
        umpire::util::Exception);

    dest_allocator->deallocate(small_dest_array);
}

const std::string zero_copy_sources[] = {
  "HOST"
};

const std::string zero_copy_dests[] = {
    "HOST"
#if defined(UMPIRE_ENABLE_DEVICE)
    , "DEVICE"
#endif
};

INSTANTIATE_TEST_CASE_P(
    ZeroCopies,
    ZeroCopyTest,
    ::testing::Combine(
      ::testing::ValuesIn(zero_copy_sources),
      ::testing::ValuesIn(zero_copy_dests)
),);

const std::string copy_sources[] = {
  "HOST"
#if defined(UMPIRE_ENABLE_UM)
  , "UM"
#endif
#if defined(UMPIRE_ENABLE_PINNED)
  , "PINNED"
#endif
};

const std::string copy_dests[] = {
    "HOST"
#if defined(UMPIRE_ENABLE_DEVICE)
    , "DEVICE"
#endif
#if defined(UMPIRE_ENABLE_UM)
    , "UM"
#endif
#if defined(UMPIRE_ENABLE_PINNED)
    , "PINNED"
#endif
};


INSTANTIATE_TEST_CASE_P(
    Copies,
    CopyTest,
    ::testing::Combine(
      ::testing::ValuesIn(copy_sources),
      ::testing::ValuesIn(copy_dests)),);

class MemsetTest :
  public OperationTest
{
};

TEST_P(MemsetTest, Memset) {
  auto& rm = umpire::ResourceManager::getInstance();

  rm.memset(source_array, 0);

  rm.copy(check_array, source_array);

  for (std::size_t i = 0; i < m_size; i++) {
    ASSERT_EQ(0, check_array[i]);
  }
}

TEST_P(MemsetTest, Offset) {
  auto& rm = umpire::ResourceManager::getInstance();

  rm.memset(source_array, 1);
  rm.memset(source_array + m_size / 2, 2);

  rm.copy(check_array, source_array);

  char * check_chars = reinterpret_cast<char*>(check_array);

  for (std::size_t i = 0; i < m_size / 2 * sizeof(float); ++i) {
    ASSERT_EQ(1, check_chars[i]);
  }

  for (std::size_t i = m_size / 2 * sizeof(float); i < m_size * sizeof(float); ++i) {
    ASSERT_EQ(2, check_chars[i]);
  }
}

TEST_P(MemsetTest, InvalidSize)
{
  auto& rm = umpire::ResourceManager::getInstance();

  ASSERT_THROW(
      rm.memset(source_array, 0, (m_size+100)*sizeof(float)),
      umpire::util::Exception);
}

TEST_P(MemsetTest, InvalidPointer)
{
  auto& rm = umpire::ResourceManager::getInstance();

  ASSERT_THROW(
    rm.memset((void*)0x1, 0),
    umpire::util::Exception);
}

const std::string memset_sources[] = {
  "HOST"
#if defined(UMPIRE_ENABLE_DEVICE)
  , "DEVICE"
#endif
#if defined(UMPIRE_ENABLE_UM)
  , "UM"
#endif
#if defined(UMPIRE_ENABLE_PINNED)
  , "PINNED"
#endif
};

const std::string memset_dests[] = {
  "HOST"
};

INSTANTIATE_TEST_CASE_P(
    Sets,
    MemsetTest,
    ::testing::Combine(
      ::testing::ValuesIn(memset_sources),
      ::testing::ValuesIn(memset_dests)),);

class ReallocateTest :
  public OperationTest
{
};

TEST_P(ReallocateTest, Reallocate)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  const std::size_t reallocated_size = (m_size/2);

  rm.memset(source_array, 0);

  source_array =
    static_cast<float*>(
        rm.reallocate(source_array, reallocated_size*sizeof(float)));

  ASSERT_EQ(
      source_allocator->getSize(source_array),
      reallocated_size*sizeof(float));

  rm.copy(check_array, source_array, reallocated_size*sizeof(float));

  for (std::size_t i = 0; i < reallocated_size; i++) {
    ASSERT_FLOAT_EQ(check_array[i], 0);
  }
}

TEST_P(ReallocateTest, ReallocateLarger)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  const std::size_t reallocated_size = (m_size+50);

  rm.memset(source_array, 1);

  source_array =
    static_cast<float*>(
        rm.reallocate(source_array, reallocated_size*sizeof(float)));

  ASSERT_EQ(
      source_allocator->getSize(source_array),
      reallocated_size*sizeof(float));

  rm.memset(source_array + m_size, 2);

  check_array =
    static_cast<float*>(
        rm.reallocate(check_array, reallocated_size*sizeof(float)));

  rm.copy(check_array,
      source_array,
      reallocated_size*sizeof(float));

  char * check_interrogator = reinterpret_cast<char*>(check_array);
  for (std::size_t i = 0; i < m_size * sizeof(float) / sizeof(char); i++) {
    ASSERT_EQ(check_interrogator[i], 1);
  }

  for (std::size_t i = m_size * sizeof(float) / sizeof(char); i < reallocated_size * sizeof(float) / sizeof(char); i++) {
    ASSERT_EQ(check_interrogator[i], 2);
  }
}

TEST_P(ReallocateTest, RealocateNull)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  rm.setDefaultAllocator(*source_allocator);

  const std::size_t reallocated_size = (m_size+50);

  void* null_array = nullptr;

  float* reallocated_array =
    static_cast<float*>(
        rm.reallocate(null_array, reallocated_size*sizeof(float)));

  ASSERT_EQ(
      source_allocator->getSize(reallocated_array),
      reallocated_size*sizeof(float));

  rm.deallocate(reallocated_array);
  rm.setDefaultAllocator(rm.getAllocator("HOST"));
}

TEST_P(ReallocateTest, ReallocateNullWithAllocator)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  const std::size_t reallocated_size = (m_size+50);

  void* null_array = nullptr;

  float* reallocated_array =
    static_cast<float*>(
        rm.reallocate(null_array, reallocated_size*sizeof(float), *source_allocator));

  ASSERT_EQ(
      source_allocator->getSize(reallocated_array),
      reallocated_size*sizeof(float));

  rm.deallocate(reallocated_array);
}

TEST_P(ReallocateTest, ReallocateWithAllocator)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  const std::size_t reallocated_size = (m_size+50);

  float* reallocated_array =
    static_cast<float*>(
        rm.reallocate(source_array, reallocated_size*sizeof(float), *source_allocator));

  ASSERT_EQ(
      source_allocator->getSize(reallocated_array),
      reallocated_size*sizeof(float));

  rm.deallocate(reallocated_array);
  source_array = nullptr;
}

TEST_P(ReallocateTest, ReallocateWithAllocatorFail)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  if (source_allocator->getId() == dest_allocator->getId()) {
    SUCCEED();
  } else {
    ASSERT_THROW(
        rm.reallocate(source_array, m_size, *dest_allocator),
        umpire::util::Exception);
  }
}

const std::string reallocate_sources[] = {
  "HOST"
#if defined(UMPIRE_ENABLE_UM)
  , "UM"
#endif
#if defined(UMPIRE_ENABLE_DEVICE)
  , "DEVICE"
#endif
#if defined(UMPIRE_ENABLE_PINNED)
  , "PINNED"
#endif
};

const std::string reallocate_dests[] = {
  "HOST"
};

INSTANTIATE_TEST_CASE_P(
    Reallocate,
    ReallocateTest,
    ::testing::Combine(
      ::testing::ValuesIn(reallocate_sources),
      ::testing::ValuesIn(reallocate_dests)),);

class MoveTest :
  public OperationTest
{
};

TEST_P(MoveTest, Move)
{
  auto& rm = umpire::ResourceManager::getInstance();

  // this works because source should always be host!
  for (std::size_t i = 0; i < m_size; i++) {
    source_array[i] = static_cast<float>(i);
  }

  float* moved_array = static_cast<float*>(rm.move(source_array, *dest_allocator));

  if ( dest_allocator->getAllocationStrategy()
      == source_allocator->getAllocationStrategy()) {
    ASSERT_EQ(moved_array, source_array);
  }

  rm.copy(check_array, moved_array);

  for (std::size_t i = 0; i < m_size; i++) {
    ASSERT_FLOAT_EQ(check_array[i], static_cast<float>(i));
  }

  dest_allocator->deallocate(moved_array);
  source_array = nullptr;
}

const std::string move_sources[] = {
  "HOST"
#if defined(UMPIRE_ENABLE_UM)
  , "UM"
#endif
#if defined(UMPIRE_ENABLE_PINNED)
  , "PINNED"
#endif
};

const std::string move_dests[] = {
  "HOST"
#if defined(UMPIRE_ENABLE_DEVICE)
  , "DEVICE"
#endif
#if defined(UMPIRE_ENABLE_UM)
  , "UM"
#endif
#if defined(UMPIRE_ENABLE_PINNED)
  , "PINNED"
#endif
};

INSTANTIATE_TEST_CASE_P(
    Move,
    MoveTest,
    ::testing::Combine(
      ::testing::ValuesIn(move_sources),
      ::testing::ValuesIn(move_dests)),);

#if defined(UMPIRE_ENABLE_CUDA)
class AdviceTest :
  public OperationTest
{
};

TEST_P(AdviceTest, ReadMostly)
{
  auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();
  auto strategy = source_allocator->getAllocationStrategy();

  int device = 0;

  auto m_advice_operation = op_registry.find(
      "READ_MOSTLY",
      strategy,
      strategy);

  if (dest_allocator->getPlatform() == umpire::Platform::cpu) {
    device = cudaCpuDeviceId;
  }

  ASSERT_NO_THROW({
      m_advice_operation->apply(
          source_array,
          nullptr,
          device,
          m_size);
  });
}

TEST_P(AdviceTest, PreferredLocation)
{
  auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();
  auto strategy = source_allocator->getAllocationStrategy();

  int device = 0;

  auto m_advice_operation = op_registry.find(
      "PREFERRED_LOCATION",
      strategy,
      strategy);

  if (dest_allocator->getPlatform() == umpire::Platform::cpu) {
    device = cudaCpuDeviceId;
  }

  ASSERT_NO_THROW({
      m_advice_operation->apply(
          source_array,
          nullptr,
          device,
          m_size);
  });
}

TEST_P(AdviceTest, AccessedBy)
{
  auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();
  auto strategy = source_allocator->getAllocationStrategy();

  int device = 0;

  auto m_advice_operation = op_registry.find(
      "ACCESSED_BY",
      strategy,
      strategy);

  if (dest_allocator->getPlatform() == umpire::Platform::cpu) {
    device = cudaCpuDeviceId;
  }

  ASSERT_NO_THROW({
      m_advice_operation->apply(
          source_array,
          nullptr,
          device,
          m_size);
  });
}

const std::string advice_sources[] = {
  "UM"
};

const std::string advice_dests[] = {
  "HOST"
  , "DEVICE"
};

INSTANTIATE_TEST_CASE_P(
    Advice,
    AdviceTest,
    ::testing::Combine(
      ::testing::ValuesIn(advice_sources),
      ::testing::ValuesIn(advice_dests)
),);

#endif

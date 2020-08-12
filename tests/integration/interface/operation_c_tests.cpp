//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/interface/umpire.h"

#if defined(UMPIRE_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif

class OperationTest : public ::testing::TestWithParam<
                          ::testing::tuple<const char*, const char*>> {
 public:
  virtual void SetUp()
  {
    umpire_resourcemanager rm;
    umpire_resourcemanager_get_instance(&rm);

    umpire_resourcemanager_get_allocator_by_name(
        &rm, ::testing::get<0>(GetParam()), &source_allocator);
    ;
    umpire_resourcemanager_get_allocator_by_name(
        &rm, ::testing::get<1>(GetParam()), &dest_allocator);
    ;

    source_array = (float*)umpire_allocator_allocate(&source_allocator,
                                                     m_size * sizeof(float));
    dest_array = (float*)umpire_allocator_allocate(&dest_allocator,
                                                   m_size * sizeof(float));

    umpire_allocator host_allocator;
    umpire_resourcemanager_get_allocator_by_name(&rm, "HOST", &host_allocator);
    ;

    check_array = (float*)umpire_allocator_allocate(&host_allocator,
                                                    m_size * sizeof(float));
  }

  virtual void TearDown()
  {
    umpire_resourcemanager rm;
    umpire_resourcemanager_get_instance(&rm);

    umpire_allocator host_allocator;
    umpire_resourcemanager_get_allocator_by_name(&rm, "HOST", &host_allocator);
    ;

    if (source_array)
      umpire_allocator_deallocate(&source_allocator, source_array);

    if (dest_array)
      umpire_allocator_deallocate(&dest_allocator, dest_array);

    if (check_array)
      umpire_allocator_deallocate(&host_allocator, check_array);

    umpire_allocator_delete(&host_allocator);
    umpire_allocator_delete(&source_allocator);
    umpire_allocator_delete(&dest_allocator);
  }

  float* source_array;
  float* dest_array;
  float* check_array;

  const std::size_t m_size = 1024;

  umpire_allocator source_allocator;
  umpire_allocator dest_allocator;
};

class CopyTest : public OperationTest {
};

TEST_P(CopyTest, Copy)
{
  umpire_resourcemanager rm;
  umpire_resourcemanager_get_instance(&rm);

  for (std::size_t i = 0; i < m_size; i++) {
    source_array[i] = i;
  }

  umpire_resourcemanager_copy_all(&rm, dest_array, source_array);
  umpire_resourcemanager_copy_all(&rm, check_array, dest_array);

  for (std::size_t i = 0; i < m_size; i++) {
    ASSERT_FLOAT_EQ(source_array[i], check_array[i]);
  }
}

TEST_P(CopyTest, CopyOffset)
{
  umpire_resourcemanager rm;
  umpire_resourcemanager_get_instance(&rm);

  check_array[10] = 3.14;

  umpire_resourcemanager_copy_with_size(&rm, &dest_array[11], &check_array[10],
                                        sizeof(float));
  umpire_resourcemanager_copy_with_size(&rm, &check_array[0], &dest_array[11],
                                        sizeof(float));

  ASSERT_EQ(check_array[0], check_array[10]);
}

const char* copy_sources[] = {"HOST"
#if defined(UMPIRE_ENABLE_UM)
                              ,
                              "UM"
#endif
#if defined(UMPIRE_ENABLE_PINNED)
                              ,
                              "PINNED"
#endif
};

const char* copy_dests[] = {"HOST"
#if defined(UMPIRE_ENABLE_DEVICE)
                            ,
                            "DEVICE"
#endif
#if defined(UMPIRE_ENABLE_UM)
                            ,
                            "UM"
#endif
#if defined(UMPIRE_ENABLE_PINNED)
                            ,
                            "PINNED"
#endif
};

INSTANTIATE_TEST_SUITE_P(Copies, CopyTest,
                         ::testing::Combine(::testing::ValuesIn(copy_sources),
                                            ::testing::ValuesIn(copy_dests)));

class MemsetTest : public OperationTest {
};

TEST_P(MemsetTest, Memset)
{
  umpire_resourcemanager rm;
  umpire_resourcemanager_get_instance(&rm);

  umpire_resourcemanager_memset_all(&rm, source_array, 0);

  umpire_resourcemanager_copy_all(&rm, check_array, source_array);

  for (std::size_t i = 0; i < m_size; i++) {
    ASSERT_EQ(0, check_array[i]);
  }
}

const char* memset_sources[] = {"HOST"
#if defined(UMPIRE_ENABLE_DEVICE)
                                ,
                                "DEVICE"
#endif
#if defined(UMPIRE_ENABLE_UM)
                                ,
                                "UM"
#endif
#if defined(UMPIRE_ENABLE_PINNED)
                                ,
                                "PINNED"
#endif
};

const char* memset_dests[] = {"HOST"};

INSTANTIATE_TEST_SUITE_P(Sets, MemsetTest,
                         ::testing::Combine(::testing::ValuesIn(memset_sources),
                                            ::testing::ValuesIn(memset_dests)));

class ReallocateTest : public OperationTest {
};

TEST_P(ReallocateTest, Reallocate)
{
  umpire_resourcemanager rm;
  umpire_resourcemanager_get_instance(&rm);

  const std::size_t reallocated_size = (m_size / 2);

  umpire_resourcemanager_memset_all(&rm, source_array, 0);

  float* reallocated_array = (float*)umpire_resourcemanager_reallocate_default(
      &rm, source_array, reallocated_size * sizeof(float));

  ASSERT_EQ(umpire_allocator_get_size(&source_allocator, reallocated_array),
            reallocated_size * sizeof(float));

  umpire_resourcemanager_copy_with_size(&rm, check_array, reallocated_array,
                                        reallocated_size * sizeof(float));

  for (std::size_t i = 0; i < reallocated_size; i++) {
    ASSERT_FLOAT_EQ(check_array[i], 0);
  }

  source_array = nullptr;
}

TEST_P(ReallocateTest, ReallocateLarger)
{
  umpire_resourcemanager rm;
  umpire_resourcemanager_get_instance(&rm);

  const std::size_t reallocated_size = (m_size + 50);

  umpire_resourcemanager_memset_all(&rm, source_array, 0);

  float* reallocated_array = (float*)umpire_resourcemanager_reallocate_default(
      &rm, source_array, reallocated_size * sizeof(float));

  ASSERT_EQ(umpire_allocator_get_size(&source_allocator, reallocated_array),
            reallocated_size * sizeof(float));

  float* reallocated_check_array =
      (float*)umpire_resourcemanager_reallocate_default(
          &rm, check_array, reallocated_size * sizeof(float));

  umpire_resourcemanager_copy_with_size(&rm, reallocated_check_array,
                                        reallocated_array,
                                        reallocated_size * sizeof(float));

  for (std::size_t i = 0; i < m_size; i++) {
    ASSERT_FLOAT_EQ(reallocated_check_array[i], 0);
  }

  umpire_resourcemanager_deallocate(&rm, reallocated_check_array);

  source_array = nullptr;
  check_array = nullptr;
}

// TEST_P(ReallocateTest, ReallocateWithAllocator)
// {
//   umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();
//
//   const std::size_t reallocated_size = (m_size+50);
//
//   float* reallocated_array =
//     static_cast<float*>(
//         rm.reallocate(source_array, reallocated_size*sizeof(float),
//         *source_allocator));
//
//   ASSERT_EQ(
//       source_allocator->getSize(reallocated_array),
//       reallocated_size*sizeof(float));
//
//   rm.deallocate(reallocated_array);
//   source_array = nullptr;
// }

const char* reallocate_sources[] = {"HOST"
#if defined(UMPIRE_ENABLE_UM)
                                    ,
                                    "UM"
#endif
#if defined(UMPIRE_ENABLE_DEVICE)
                                    ,
                                    "DEVICE"
#endif
#if defined(UMPIRE_ENABLE_PINNED)
                                    ,
                                    "PINNED"
#endif
};

const char* reallocate_dests[] = {"HOST"};

INSTANTIATE_TEST_SUITE_P(
    Reallocate, ReallocateTest,
    ::testing::Combine(::testing::ValuesIn(reallocate_sources),
                       ::testing::ValuesIn(reallocate_dests)));

// class MoveTest :
//   public OperationTest
// {
// };
//
// TEST_P(MoveTest, Move)
// {
//   auto& rm = umpire::ResourceManager::getInstance();
//
//   // this works because source should always be host!
//   for (std::size_t i = 0; i < m_size; i++) {
//     source_array[i] = i;
//   }
//
//   float* moved_array = static_cast<float*>(rm.move(source_array,
//   *dest_allocator));
//
//   if ( dest_allocator->getAllocationStrategy()
//       == source_allocator->getAllocationStrategy()) {
//     ASSERT_EQ(moved_array, source_array);
//   }
//
//   rm.copy(check_array, moved_array);
//
//   for (std::size_t i = 0; i < m_size; i++) {
//     ASSERT_FLOAT_EQ(check_array[i], i);
//   }
//
//   dest_allocator->deallocate(moved_array);
//   source_array = nullptr;
// }
//
// const std::string move_sources[] = {
//   "HOST"
// #if defined(UMPIRE_ENABLE_UM)
//   , "UM"
// #endif
// #if defined(UMPIRE_ENABLE_PINNED)
//   , "PINNED"
// #endif
// };
//
// const std::string move_dests[] = {
//   "HOST"
// #if defined(UMPIRE_ENABLE_DEVICE)
//   , "DEVICE"
// #endif
// #if defined(UMPIRE_ENABLE_UM)
//   , "UM"
// #endif
// #if defined(UMPIRE_ENABLE_PINNED)
//   , "PINNED"
// #endif
// };
//
// INSTANTIATE_TEST_CASE_P(
//     Move,
//     MoveTest,
//     ::testing::Combine(
//       ::testing::ValuesIn(move_sources),
//       ::testing::ValuesIn(move_dests)
// ),);
//
// #if defined(UMPIRE_ENABLE_CUDA)
// class AdviceTest :
//   public OperationTest
// {
// };
//
// TEST_P(AdviceTest, ReadMostly)
// {
//   auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();
//   auto strategy = source_allocator->getAllocationStrategy();
//
//   int device = 0;
//
//   auto m_advice_operation = op_registry.find(
//       "READ_MOSTLY",
//       strategy,
//       strategy);
//
//   if (dest_allocator->getPlatform() == umpire::Platform::host) {
//     device = cudaCpuDeviceId;
//   }
//
//   ASSERT_NO_THROW({
//       m_advice_operation->apply(
//           source_array,
//           nullptr,
//           device,
//           m_size);
//   });
// }
//
// TEST_P(AdviceTest, PreferredLocation)
// {
//   auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();
//   auto strategy = source_allocator->getAllocationStrategy();
//
//   int device = 0;
//
//   auto m_advice_operation = op_registry.find(
//       "PREFERRED_LOCATION",
//       strategy,
//       strategy);
//
//   if (dest_allocator->getPlatform() == umpire::Platform::host) {
//     device = cudaCpuDeviceId;
//   }
//
//   ASSERT_NO_THROW({
//       m_advice_operation->apply(
//           source_array,
//           nullptr,
//           device,
//           m_size);
//   });
// }
//
// TEST_P(AdviceTest, AccessedBy)
// {
//   auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();
//   auto strategy = source_allocator->getAllocationStrategy();
//
//   int device = 0;
//
//   auto m_advice_operation = op_registry.find(
//       "ACCESSED_BY",
//       strategy,
//       strategy);
//
//   if (dest_allocator->getPlatform() == umpire::Platform::host) {
//     device = cudaCpuDeviceId;
//   }
//
//   ASSERT_NO_THROW({
//       m_advice_operation->apply(
//           source_array,
//           nullptr,
//           device,
//           m_size);
//   });
// }
//
// const std::string advice_sources[] = {
//   "UM"
// };
//
// const std::string advice_dests[] = {
//   "HOST"
//   , "DEVICE"
// };
//
// INSTANTIATE_TEST_CASE_P(
//     Advice,
//     AdviceTest,
//     ::testing::Combine(
//       ::testing::ValuesIn(advice_sources),
//       ::testing::ValuesIn(advice_dests)
// ),);
//
// #endif

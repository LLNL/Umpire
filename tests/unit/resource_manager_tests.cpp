//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/NamedAllocationStrategy.hpp"

TEST(ResourceManager, Constructor)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  (void)rm;
  SUCCEED();
}

TEST(ResourceManager, findAllocationRecord)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc = rm.getAllocator("HOST");

  const std::size_t size = 1024 * 1024;
  const std::size_t offset = 1024;

  char* ptr = static_cast<char*>(alloc.allocate(size));
  const umpire::util::AllocationRecord* rec_begin = rm.findAllocationRecord(ptr);
  const umpire::util::AllocationRecord* rec_middle = rm.findAllocationRecord(ptr + offset);
  const umpire::util::AllocationRecord* rec_end = rm.findAllocationRecord(ptr + (size - 1));

  ASSERT_EQ(ptr, rec_begin->ptr);
  ASSERT_EQ(ptr, rec_middle->ptr);
  ASSERT_EQ(ptr, rec_end->ptr);

  ASSERT_THROW(rm.findAllocationRecord(ptr + size), umpire::runtime_error);

  ASSERT_THROW(rm.findAllocationRecord(ptr + size + 1), umpire::runtime_error);

  alloc.deallocate(ptr);

  ASSERT_THROW(rm.findAllocationRecord(nullptr), umpire::runtime_error);
}

TEST(ResourceManager, getAllocatorByName)
{
  auto& rm = umpire::ResourceManager::getInstance();

  EXPECT_NO_THROW({
    auto alloc = rm.getAllocator("HOST");
    UMPIRE_USE_VAR(alloc);
  });

  ASSERT_THROW(rm.getAllocator("BANANA"), umpire::runtime_error);
}

TEST(ResourceManager, getAllocatorById)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc = rm.getAllocator("HOST");
  int id = alloc.getId();

  EXPECT_NO_THROW({
    alloc = rm.getAllocator(id);
    UMPIRE_USE_VAR(alloc);
  });

  ASSERT_THROW(rm.getAllocator(-4), umpire::runtime_error);
}

TEST(ResourceManager, getAllocatorInvalidId)
{
  auto& rm = umpire::ResourceManager::getInstance();

  ASSERT_THROW(rm.getAllocator(umpire::invalid_allocator_id), umpire::runtime_error);
}

TEST(ResourceManager, aliases)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc = rm.getAllocator("HOST");
  rm.addAlias("HOST_ALIAS", alloc);
  EXPECT_THROW({ rm.addAlias("HOST_ALIAS", alloc); }, umpire::runtime_error);

  auto alloc_alias = rm.getAllocator("HOST_ALIAS");
  EXPECT_EQ(alloc_alias.getId(), alloc.getId());
  EXPECT_EQ(alloc_alias.getName(), alloc.getName());
  EXPECT_NE(alloc_alias.getName(), "HOST_ALIAS");

  EXPECT_THROW({ rm.removeAlias("BLAH", alloc); }, umpire::runtime_error);
  EXPECT_NO_THROW({ rm.removeAlias("HOST_ALIAS", alloc); });

  alloc = rm.getAllocator("HOST");
  EXPECT_THROW({ rm.removeAlias("HOST", alloc); }, umpire::runtime_error);

  auto named_alloc = rm.makeAllocator<umpire::strategy::NamedAllocationStrategy>("NAMED_ALLOCATOR", alloc);

  EXPECT_THROW({ rm.removeAlias("NAMED_ALLOCATOR", named_alloc); }, umpire::runtime_error);
}
/*
#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)
TEST(ResourceManager, DeviceMemsetDeviceAllocator)
{
  auto& rm = umpire::ResourceManager::getInstance();

  if (!rm.isAllocator("DEVICE")) {
    GTEST_SKIP() << "DEVICE allocator not available in this build.";
  }

  auto device_alloc = rm.getAllocator("DEVICE");
  auto host_alloc = rm.getAllocator("HOST");

  constexpr std::size_t n = 256;
  constexpr double value = 3.14;

  double* d_ptr = static_cast<double*>(device_alloc.allocate(n * sizeof(double)));
  double* h_ptr = static_cast<double*>(host_alloc.allocate(n * sizeof(double)));

  rm.deviceMemset(d_ptr, n, value);
  rm.copy(h_ptr, d_ptr, n * sizeof(double));

  for (std::size_t i = 0; i < n; ++i) {
    ASSERT_EQ(h_ptr[i], value);
  }

  device_alloc.deallocate(d_ptr);
  host_alloc.deallocate(h_ptr);
}
#endif
*/

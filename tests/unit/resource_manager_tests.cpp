//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
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
  const umpire::util::AllocationRecord* rec_begin =
      rm.findAllocationRecord(ptr);
  const umpire::util::AllocationRecord* rec_middle =
      rm.findAllocationRecord(ptr + offset);
  const umpire::util::AllocationRecord* rec_end =
      rm.findAllocationRecord(ptr + (size - 1));

  ASSERT_EQ(ptr, rec_begin->ptr);
  ASSERT_EQ(ptr, rec_middle->ptr);
  ASSERT_EQ(ptr, rec_end->ptr);

  ASSERT_THROW(rm.findAllocationRecord(ptr + size), umpire::util::Exception);

  ASSERT_THROW(rm.findAllocationRecord(ptr + size + 1),
               umpire::util::Exception);

  alloc.deallocate(ptr);

  ASSERT_THROW(rm.findAllocationRecord(nullptr), umpire::util::Exception);
}

TEST(ResourceManager, getAllocatorByName)
{
  auto& rm = umpire::ResourceManager::getInstance();

  EXPECT_NO_THROW({
    auto alloc = rm.getAllocator("HOST");
    UMPIRE_USE_VAR(alloc);
  });

  ASSERT_THROW(rm.getAllocator("BANANA"), umpire::util::Exception);
}

TEST(ResourceManager, getAllocatorById)
{
  auto& rm = umpire::ResourceManager::getInstance();

  EXPECT_NO_THROW({
    auto alloc = rm.getAllocator(0);
    UMPIRE_USE_VAR(alloc);
  });

  ASSERT_THROW(rm.getAllocator(-4), umpire::util::Exception);
}

TEST(ResourceManager, getAllocatorInvalidId)
{
  auto& rm = umpire::ResourceManager::getInstance();

  ASSERT_THROW(rm.getAllocator(umpire::invalid_allocator_id),
               umpire::util::Exception);
}

TEST(ResourceManager, aliases)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc = rm.getAllocator("HOST");
  rm.addAlias("HOST_ALIAS", alloc);
  EXPECT_THROW({
      rm.addAlias("HOST_ALIAS", alloc);
  }, umpire::util::Exception);

  auto alloc_alias = rm.getAllocator("HOST_ALIAS");
  EXPECT_EQ(alloc_alias.getId(), alloc.getId());
  EXPECT_EQ(alloc_alias.getName(), alloc.getName());
  EXPECT_NE(alloc_alias.getName(), "HOST_ALIAS");

  EXPECT_THROW({
    rm.removeAlias("BLAH", alloc);
  }, umpire::util::Exception);
  EXPECT_NO_THROW({
    rm.removeAlias("HOST_ALIAS", alloc);
  });

  alloc = rm.getAllocator("HOST");
  EXPECT_THROW({
    rm.removeAlias("HOST", alloc);
  }, umpire::util::Exception);

  auto named_alloc = rm.makeAllocator<umpire::strategy::NamedAllocationStrategy>(
    "NAMED_ALLOCATOR", alloc);

  EXPECT_THROW({
    rm.removeAlias("NAMED_ALLOCATOR", named_alloc);
  }, umpire::util::Exception);
}
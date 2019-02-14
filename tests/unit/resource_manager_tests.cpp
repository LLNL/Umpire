//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"

#include "umpire/ResourceManager.hpp"

TEST(ResourceManager, Constructor) {
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  (void) rm;
  SUCCEED();
}

TEST(ResourceManager, findAllocationRecord)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc = rm.getAllocator("HOST");

  const size_t size = 10 * sizeof(double);
  const size_t offset = 2 * sizeof(double); // < size

  double* ptr = static_cast<double*>(alloc.allocate(size));
  const umpire::util::AllocationRecord* rec = rm.findAllocationRecord(ptr + offset);

  ASSERT_THROW(rm.findAllocationRecord(nullptr), umpire::util::Exception);

  ASSERT_EQ(ptr, rec->m_ptr);

  alloc.deallocate(ptr);
}

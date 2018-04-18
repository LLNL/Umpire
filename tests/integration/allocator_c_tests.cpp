//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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

#include "umpire/umpire.h"

TEST(Allocator, HostAllocator)
{
  UMPIRE_resourcemanager* rm = UMPIRE_resourcemanager_get();
  UMPIRE_allocator* allocator = UMPIRE_resourcemanager_get_allocator(rm, "HOST");

  double* test_alloc = (double*) UMPIRE_allocator_allocate(allocator, 100*sizeof(double));

  ASSERT_NE(nullptr, test_alloc);
  UMPIRE_allocator_deallocate(allocator, test_alloc);
}

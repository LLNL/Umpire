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
#include <iostream>
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/SlotPool.hpp"
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/op/MemoryOperation.hpp"
#include "umpire/strategy/AllocationAdvisor.hpp"

int main(int , char** )
{
  std::cout << "Hello from your friendly cpp test program!\n";
  return 0;
}

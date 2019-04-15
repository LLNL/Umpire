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

#include "umpire/ResourceManager.hpp"

#include <iostream>

namespace umpire {

void print_allocator_records(Allocator alloc, std::ostream& os) {
  auto& rm = umpire::ResourceManager::getInstance();

  auto strategy = alloc.getAllocationStrategy();

  rm.m_allocations.print([strategy] (const util::AllocationRecord* rec) {
    return rec->strategy == strategy;
  }, os);
}

}

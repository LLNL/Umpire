//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/config.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/ResourceManager.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>

volatile int umpire_ver_2_found;

namespace umpire {

void print_allocator_records(Allocator allocator, std::ostream& os)
{
  std::stringstream ss;
  auto& rm = umpire::ResourceManager::getInstance();

  auto strategy = allocator.getAllocationStrategy();

  rm.m_allocations.print([strategy] (const util::AllocationRecord& rec) {
    return rec.strategy == strategy;
  }, ss);

  if (! ss.str().empty() ) {
    os << "Allocations for "
      << allocator.getName()
      << " allocator:" << std::endl
      << ss.str() << std::endl;
  }
}

std::vector<util::AllocationRecord> get_allocator_records(Allocator allocator)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto strategy = allocator.getAllocationStrategy();

  std::vector<util::AllocationRecord> recs;
  std::copy_if(rm.m_allocations.begin(), rm.m_allocations.end(),
               std::back_inserter(recs), [strategy] (const util::AllocationRecord& rec) {
                 return rec.strategy == strategy;
               });

  return recs;
}

} // end namespace umpire

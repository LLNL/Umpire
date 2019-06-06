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
#include "umpire/strategy/mixins/Inspector.hpp"

#include "umpire/ResourceManager.hpp"

namespace umpire {
namespace strategy {
namespace mixins {

Inspector::Inspector() :
  m_current_size(0),
  m_high_watermark(0)
{
}


void
Inspector::registerAllocation(
    void* ptr,
    size_t size,
    strategy::AllocationStrategy* strategy)
{
  m_current_size += size;

  if (m_current_size > m_high_watermark) {
    m_high_watermark = m_current_size;
  }

  ResourceManager::getInstance().registerAllocation(ptr, {ptr, size, strategy});
}

util::AllocationRecord
Inspector::deregisterAllocation(void* ptr, strategy::AllocationStrategy* strategy)
{
  auto record = ResourceManager::getInstance().deregisterAllocation(ptr);

  if (record.strategy == strategy) {
    m_current_size -= record.size;
  } else {
    // Re-register the pointer and throw an error
    ResourceManager::getInstance().registerAllocation(ptr, {ptr, record.size, record.strategy});
    UMPIRE_ERROR(ptr << " was not allocated by " << strategy->getName());
  }

  return record;
}

} // end of namespace mixins
} // end of namespace strategy
} // end of namespace umpire

//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/mixins/Inspector.hpp"

#include "umpire/ResourceManager.hpp"

namespace umpire {
namespace strategy {
namespace mixins {

Inspector::Inspector(AllocationStrategy* s) :
  m_strategy{s},
  m_strategy_tracks_use{s->tracksMemoryUse()}
{
}

void
Inspector::registerAllocation(
    void* ptr,
    std::size_t size,
    strategy::AllocationStrategy*) 
{
  if (! m_strategy_tracks_use) {
    m_strategy->m_current_size += size;
    m_strategy->m_allocation_count++;

    if (m_strategy->m_current_size > m_strategy->m_high_watermark) {
      m_strategy->m_high_watermark = m_strategy->m_current_size;
    }
  }

  ResourceManager::getInstance().registerAllocation(ptr, {ptr, size, m_strategy});
}

util::AllocationRecord
Inspector::deregisterAllocation(void* ptr, strategy::AllocationStrategy* strategy)
{
  auto record = ResourceManager::getInstance().deregisterAllocation(ptr);

  if (record.strategy == m_strategy) {
    if (! m_strategy_tracks_use) {
      m_strategy->m_current_size -= record.size;
      m_strategy->m_allocation_count--;
    }
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

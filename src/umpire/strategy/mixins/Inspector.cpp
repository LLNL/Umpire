//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/mixins/Inspector.hpp"

#include "umpire/ResourceManager.hpp"

#include <string>

namespace umpire {
namespace strategy {
namespace mixins {

void Inspector::registerAllocation(void* ptr, std::size_t size, strategy::AllocationStrategy* s)
{
  s->m_current_size += size;
  s->m_allocation_count++;

  if (s->m_current_size > s->m_high_watermark) {
    s->m_high_watermark = s->m_current_size;
  }

  ResourceManager::getInstance().registerAllocation(ptr, {ptr, size, s});
}

void Inspector::registerAllocation(void* ptr, std::size_t size, strategy::AllocationStrategy* s, const std::string& name)
{
  s->m_current_size += size;
  s->m_allocation_count++;

  if (s->m_current_size > s->m_high_watermark) {
    s->m_high_watermark = s->m_current_size;
  }

  ResourceManager::getInstance().registerAllocation(ptr, {ptr, size, s, name});
}

util::AllocationRecord
Inspector::deregisterAllocation(void* ptr, strategy::AllocationStrategy* s)
{
  auto record = ResourceManager::getInstance().deregisterAllocation(ptr);

  if (record.strategy == s) {
    s->m_current_size -= record.size;
    s->m_allocation_count--;
  } else {
    // Re-register the pointer and throw an error
    ResourceManager::getInstance().registerAllocation(ptr, {ptr, record.size, record.strategy, record.name});
    UMPIRE_ERROR(runtime_error, fmt::format("{} was not allocated by {}", ptr, s->getName()));
  }

  return record;
}

} // end of namespace mixins
} // end of namespace strategy
} // end of namespace umpire

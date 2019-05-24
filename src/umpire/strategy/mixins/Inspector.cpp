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

#include "umpire/util/AllocationRecord.hpp"

#include "umpire/ResourceManager.hpp"

#include "umpire/util/FixedMallocPool.hpp"

namespace {
static umpire::util::FixedMallocPool pool(sizeof(umpire::util::AllocationRecord));
}

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

  auto record = new (pool.allocate()) umpire::util::AllocationRecord{
    ptr,
    size,
    strategy};

  ResourceManager::getInstance().registerAllocation(ptr, record);
}

util::AllocationRecord
Inspector::deregisterAllocation(void* ptr)
{
  auto record = ResourceManager::getInstance().deregisterAllocation(ptr);

  m_current_size -= record->m_size;

  util::AllocationRecord rec(*record);
  pool.deallocate(record);
  //delete record;
  return rec;
}

} // end of namespace mixins
} // end of namespace strategy
} // end of namespace umpire

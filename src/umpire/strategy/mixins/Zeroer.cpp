//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/mixins/Zeroer.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/FixedPool.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {
namespace mixins {

Zeroer::Zeroer() :
  m_zero_byte_pool{nullptr}
{
}

void* Zeroer::allocateZero()
{
  if (!m_zero_byte_pool)
    m_zero_byte_pool = static_cast<FixedPool*>(
          ResourceManager::getInstance().getZeroByteAllocator());

  return m_zero_byte_pool->allocate(1);
}

bool Zeroer::deallocateZero(void* ptr)
{
  if (!m_zero_byte_pool)
    m_zero_byte_pool = static_cast<FixedPool*>(
          ResourceManager::getInstance().getZeroByteAllocator());

  if (m_zero_byte_pool->pointerIsFromPool(ptr)) {
    m_zero_byte_pool->deallocate(ptr, 1);
    return true;
  } else {
    return false;
  }
}

} // end of namespace mixins
} // end of namespace strategy
} // end of namespace umpire
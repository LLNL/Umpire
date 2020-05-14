//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/SharedMemoryAllocator.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/Replay.hpp"

#include <string>

namespace umpire {

SharedMemoryAllocator::SharedMemoryAllocator(strategy::SharedMemoryAllocation* allocator) noexcept:
  m_allocator(allocator)
{
}

void* SharedMemoryAllocator::allocate(std::size_t bytes)
{
  void* ret = nullptr;

  UMPIRE_LOG(Debug, "(" << bytes << ")");

  UMPIRE_REPLAY(
    R"("event": "allocate", "payload": {)"
    << R"( "allocator_ref": ")" << m_allocator
    << R"(", "size": )" << bytes << " }");

  ret = m_allocator->allocate(bytes);

  UMPIRE_REPLAY(
    R"("event": "allocate", "payload": {)"
    << R"( "allocator_ref": ")" << m_allocator
    << R"(", "size": )" << bytes << " }"
    << R"(", "result": { "memory_ptr": ")" << ret << R"("" })");

  // UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ret), "size", bytes, "event", "allocate");
  return ret;
}

void* SharedMemoryAllocator::allocate(std::string name, std::size_t bytes)
{
  void* ret = nullptr;

  UMPIRE_LOG(Debug, "(" << bytes << ")");

  UMPIRE_REPLAY(
    R"("event": "named_allocate", "payload": {)"
    << R"( "allocator_ref": ")" << m_allocator << R"(",)"
    << R"( "name": ")" << name << R"(",)"
    << R"( "size": )" << bytes << " }");

  ret = m_allocator->allocate(bytes);

  UMPIRE_REPLAY(
    R"("event": "named_allocate", "payload": {)"
    << R"( "allocator_ref": ")" << m_allocator << R"(",)"
    << R"( "name": ")" << name << R"(",)"
    << R"( "size": )" << bytes << " }"
    R"(, "result": { "memory_ptr": ")" << ret << R"("" })");

  // UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ret), "size", bytes, "event", "allocate");
  return ret;
}

void* SharedMemoryAllocator::get_allocation_by_name(std::string name)
{
  UMPIRE_LOG(Debug, "(" << name << ")");
  return m_allocator->get_allocation_by_name(name);
}


void SharedMemoryAllocator::deallocate(void* ptr)
{
  UMPIRE_REPLAY(
    R"("event": "deallocate", "payload": {)"
    << R"("allocator_ref": ")" << m_allocator << R"(",)"
    << R"( "memory_ptr": ")" << ptr << R"(" })");

  UMPIRE_LOG(Debug, "(" << ptr << ")");

  //UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr), "size", 0x0, "event", "deallocate");

  if (!ptr) {
    UMPIRE_LOG(Info, "Deallocating a null pointer");
    return;
  } else {
    m_allocator->deallocate(ptr);
  }
}

void SharedMemoryAllocator::release()
{
  UMPIRE_REPLAY(R"("event": "release", "payload": { "allocator_ref": ")" <<  m_allocator << R"(" })");

  UMPIRE_LOG(Debug, "");

  m_allocator->release();
}

std::size_t SharedMemoryAllocator::getSize(void* ptr) const
{
  UMPIRE_LOG(Debug, "(" << ptr << ")");
  return ResourceManager::getInstance().getSize(ptr);
}

std::size_t SharedMemoryAllocator::getHighWatermark() const noexcept
{
  return m_allocator->getHighWatermark();
}

std::size_t SharedMemoryAllocator::getCurrentSize() const noexcept
{
  return m_allocator->getCurrentSize();
}

std::size_t SharedMemoryAllocator::getActualSize() const noexcept
{
    return m_allocator->getActualSize();
}

std::size_t SharedMemoryAllocator::getAllocationCount() const noexcept
{
  return m_allocator->getAllocationCount();
}

const std::string& SharedMemoryAllocator::getName() const noexcept
{
  return m_allocator->getName();
}

int SharedMemoryAllocator::getId() const noexcept
{
  return m_allocator->getId();
}

/**
strategy::SharedMemoryAllocation* SharedMemoryAllocator::getAllocationStrategy() noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_allocator);
  return m_allocator;
}
**/

Platform SharedMemoryAllocator::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

void SharedMemoryAllocator::set_foreman(int id)
{
  UMPIRE_LOG(Debug, "(" << id << ")");
  m_allocator->set_foreman(id);
}

bool SharedMemoryAllocator::is_foreman()
{
  UMPIRE_LOG(Debug, "()" );
  return m_allocator->is_foreman();
}

void SharedMemoryAllocator::synchronize()
{
  UMPIRE_LOG(Debug, "()" );
  m_allocator->synchronize();
}

std::ostream& operator<<(std::ostream& os, const SharedMemoryAllocator& allocator) {
    os << *allocator.m_allocator;
    return os;
}

} // end of namespace umpire

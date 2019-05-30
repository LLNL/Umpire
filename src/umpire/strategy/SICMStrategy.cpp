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
#include "umpire/strategy/SICMStrategy.hpp"
#include "umpire/util/Macros.hpp"

#include "umpire/ResourceManager.hpp"

#include <algorithm>

namespace umpire {

namespace strategy {

sicm_device_list SICMStrategy::m_devices = sicm_init();

SICMStrategy::SICMStrategy(
    const std::string& name,
    int id,
    size_t device_index,
    size_t max_size) :
  AllocationStrategy(name, id),
  m_index(device_index),
  m_max_size(max_size),
  m_arena(nullptr)
{
  if (device_index >= m_devices.count) {
    sicm_fini();
    UMPIRE_ERROR("SICMStrategy error: Device index too large");
  }

  m_arena = sicm_arena_create(m_max_size, &m_devices.devices[m_index]);
}

SICMStrategy::~SICMStrategy() {
    sicm_arena_destroy(m_arena);
    sicm_fini();
}

void*
SICMStrategy::allocate(size_t bytes)
{
  void* ret = sicm_arena_alloc(m_arena, bytes);

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

  return ret;
}

void
SICMStrategy::deallocate(void* ptr)
{
  sicm_free(ptr);
}

long
SICMStrategy::getCurrentSize() const noexcept
{
  return sicm_arena_size(m_arena);
}

long
SICMStrategy::getHighWatermark() const noexcept
{
  return m_max_size?m_max_size:sicm_arena_size(m_arena);
}

Platform
SICMStrategy::getPlatform() noexcept
{
    return Platform::sicm;
}

size_t
SICMStrategy::getDeviceIndex() const noexcept
{
  return m_index;
}

} // end of namespace strategy
} // end of namespace umpire

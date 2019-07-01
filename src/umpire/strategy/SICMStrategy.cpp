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
    sicm_device_list devices,
    std::size_t max_size) :
  AllocationStrategy(name, id),
  m_max_size(max_size),
  m_arena(nullptr)
{
  if (!devices.count || !devices.devices) {
    sicm_fini();
    UMPIRE_ERROR("SICMStrategy error: No devices specified");
  }

  m_arena = sicm_arena_create(m_max_size, static_cast<sicm_arena_flags>(0), &devices);
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

std::size_t
SICMStrategy::getCurrentSize() const noexcept
{
  return sicm_arena_size(m_arena);
}

std::size_t
SICMStrategy::getHighWatermark() const noexcept
{
  return m_max_size?m_max_size:sicm_arena_size(m_arena);
}

Platform
SICMStrategy::getPlatform() noexcept
{
  return Platform::sicm;
}

sicm_arena
SICMStrategy::getArena() const noexcept
{
  return m_arena;
}

} // end of namespace strategy
} // end of namespace umpire

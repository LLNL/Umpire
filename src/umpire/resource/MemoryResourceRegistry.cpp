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
#include "umpire/resource/MemoryResourceRegistry.hpp"

#include "umpire/util/Macros.hpp"
#include "umpire/Replay.hpp"

namespace umpire {
namespace resource {

MemoryResourceRegistry* MemoryResourceRegistry::s_allocator_registry_instance = nullptr;

MemoryResourceRegistry&
MemoryResourceRegistry::getInstance() noexcept
{
  if (!s_allocator_registry_instance) {
    s_allocator_registry_instance = new MemoryResourceRegistry();
  }

  return *s_allocator_registry_instance;
}

MemoryResourceRegistry::MemoryResourceRegistry() noexcept :
  m_allocator_factories()
{
}

void
MemoryResourceRegistry::registerMemoryResource(MemoryResourceFactory* factory)
{
  m_allocator_factories.push_front(factory);
}

resource::MemoryResource*
MemoryResourceRegistry::makeMemoryResource(const std::string& name, int id)
{
  for (auto allocator_factory : m_allocator_factories) {
    if (allocator_factory->isValidMemoryResourceFor(name)) {
      auto a = allocator_factory->create(name, id);
      UMPIRE_REPLAY("makeMemoryResource," << name << "," << a << "\n");
      return a;
    }
  }

  UMPIRE_ERROR("MemoryResource " << name << " not found");
}

} // end of namespace resource
} // end of namespace umpire

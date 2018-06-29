//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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

namespace umpire {
namespace resource {

MemoryResourceRegistry* MemoryResourceRegistry::s_allocator_registry_instance = nullptr;

MemoryResourceRegistry&
MemoryResourceRegistry::getInstance()
{
  if (!s_allocator_registry_instance) {
    s_allocator_registry_instance = new MemoryResourceRegistry();
  }

  return *s_allocator_registry_instance;
}

MemoryResourceRegistry::MemoryResourceRegistry() :
  m_allocator_factories()
{
}

void
MemoryResourceRegistry::registerMemoryResource(std::shared_ptr<MemoryResourceFactory>&& factory)
{
  m_allocator_factories.push_front(factory);
}

std::shared_ptr<umpire::resource::MemoryResource>
MemoryResourceRegistry::makeMemoryResource(const std::string& name, int id)
{
  for (auto allocator_factory : m_allocator_factories) {
    if (allocator_factory->isValidMemoryResourceFor(name)) {
        return allocator_factory->create(name, id);
    }
  }

  UMPIRE_ERROR("MemoryResource " << name << " not found");
}

} // end of namespace resource
} // end of namespace umpire

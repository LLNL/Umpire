//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/MemoryResourceRegistry.hpp"

#include "umpire/util/Macros.hpp"
#include "umpire/Replay.hpp"

namespace umpire {
namespace resource {

MemoryResourceRegistry&
MemoryResourceRegistry::getInstance() noexcept
{
  static MemoryResourceRegistry resource_registry;

  return resource_registry;
}

MemoryResourceRegistry::MemoryResourceRegistry() noexcept :
  m_allocator_factories()
{
}

void
MemoryResourceRegistry::registerMemoryResource(std::unique_ptr<MemoryResourceFactory>&& factory)
{
  m_allocator_factories.push_back(std::move(factory));
}

std::unique_ptr<resource::MemoryResource>
MemoryResourceRegistry::makeMemoryResource(const std::string& name, int id)
{
  for (auto const& allocator_factory : m_allocator_factories) {
    if (allocator_factory->isValidMemoryResourceFor(name)) {
      auto a = allocator_factory->create(name, id);
      return a;
    }
  }

  UMPIRE_ERROR("MemoryResource " << name << " not found");
}

} // end of namespace resource
} // end of namespace umpire

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
      UMPIRE_REPLAY(
           "\"event\": \"makeMemoryResource\""
        << ", \"payload\": { \"name\": \"" << name << "\" }"
        << ", \"result\": \"" << a.get() << "\""
      );
      return a;
    }
  }

  UMPIRE_ERROR("MemoryResource " << name << " not found");
}

} // end of namespace resource
} // end of namespace umpire

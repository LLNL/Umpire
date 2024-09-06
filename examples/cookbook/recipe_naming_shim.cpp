//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"
#include "umpire/resource/HostSharedMemoryResource.hpp"
#include "umpire/strategy/NamingShim.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto traits{umpire::get_default_resource_traits("SHARED")};
  traits.size = 1 * 1024 * 1024; // Maximum size of this Allocator
  traits.scope = umpire::MemoryResourceTraits::shared_scope::node; // default
  auto node_allocator{rm.makeResource("SHARED::node_allocator", traits)};
  auto shim{rm.makeAllocator<umpire::strategy::NamingShim>("shim", node_allocator)};

  void* ptr = shim.allocate(1024);
  std::cout << "Ptr = " << ptr << std::endl;
  shim.deallocate(ptr);
}

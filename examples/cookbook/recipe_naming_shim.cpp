//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
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
#include "umpire/util/shared_memory_helper.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto traits{umpire::get_default_resource_traits("SHARED::POSIX")};
  traits.size = 1 * 1024 * 1024;                                   // Maximum size of this Allocator
  traits.scope = umpire::MemoryResourceTraits::shared_scope::node; // default

  auto node_allocator{rm.makeResource("SHARED::POSIX::ipc_node_allocator", traits)};

  // Use the matchesSharedMemoryResource function to double check the type of shared memory
  if (umpire::util::matchesSharedMemoryResource("SHARED::POSIX::ipc_node_allocator", "POSIX")) {
    std::cout << "Confirmed this is an IPC Shared Memory allocator..." <<std::endl;
  }

  auto shim{rm.makeAllocator<umpire::strategy::NamingShim>("shim", node_allocator)};

  void* ptr = shim.allocate(1024);
  std::cout << "Ptr = " << ptr << std::endl;
  shim.deallocate(ptr);

  return 0;
}

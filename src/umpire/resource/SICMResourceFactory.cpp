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
#include "umpire/resource/SICMResourceFactory.hpp"

#include "umpire/resource/DefaultMemoryResource.hpp"

#include "umpire/alloc/SICMAllocator.hpp"

#include "umpire/util/detect_vendor.hpp"
#include "umpire/util/make_unique.hpp"

#include <numa.h>

#include <sicm_low.h>

namespace umpire {
namespace resource {

SICMResourceFactory::SICMResourceFactory(const std::string& name, const std::shared_ptr<sicm_device_list>& devices)
  : replacement(name),
    devices(devices)
{
}

bool
SICMResourceFactory::isValidMemoryResourceFor(const std::string& name) noexcept
{
  return (replacement.compare(name) == 0);
}

std::unique_ptr<resource::MemoryResource>
SICMResourceFactory::create(const std::string& name, int id)
{
  MemoryResourceTraits traits;

  traits.unified = true;
  traits.size = 0;

  const sicm_device_list devs = sicm_init();
  for(unsigned int i = 0; i < devs.count; i += 3) {
    traits.size += numa_node_size(devs.devices[i]->node, nullptr);
  }
  sicm_fini();

  traits.vendor = cpu_vendor_type();
  traits.kind = MemoryResourceTraits::memory_type::UNKNOWN;
  traits.used_for = MemoryResourceTraits::optimized_for::any;

  return util::make_unique<DefaultMemoryResource<alloc::SICMAllocator>>(Platform::sicm, name, id, traits, name, devices.get());
}

} // end of namespace resource
} // end of namespace umpire

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
#include "umpire/resource/MemoryResourceTypes.hpp"
#include "umpire/resource/NumaMemoryResourceFactory.hpp"
#include "umpire/resource/NumaMemoryResource.hpp"
#include "umpire/resource/DetectVendor.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace resource {

NumaMemoryResourceFactory::NumaMemoryResourceFactory(const int numa_node_)
  : m_numa_node(numa_node_),
    m_preferred_node(numa::preferred_node()),
    m_node_type(numa::node_type(m_numa_node))
{
}

bool
NumaMemoryResourceFactory::isValidMemoryResourceFor(const std::string& name,
                                                    const MemoryResourceTraits traits)
  noexcept
{
  const bool valid_for_host = ((name.compare(type_to_string(resource::Host)) == 0) &&
                               (m_numa_node == m_preferred_node));
  const bool valid_for_other = ((traits.numa_node == m_numa_node) &&
                                (m_node_type == numa::NodeType::Host));
  return valid_for_host || valid_for_other;
}

std::shared_ptr<MemoryResource>
NumaMemoryResourceFactory::create(const std::string& name, int id)
{
  MemoryResourceTraits traits;

  traits.unified = false;
  traits.numa_node = m_numa_node;

  traits.vendor = cpu_vendor_type();

  return std::make_shared<resource::NumaMemoryResource >(name, id, traits);
}

} // end of namespace resource
} // end of namespace umpire

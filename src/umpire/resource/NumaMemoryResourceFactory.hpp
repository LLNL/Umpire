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
#ifndef UMPIRE_NumaMemoryResourceFactory_HPP
#define UMPIRE_NumaMemoryResourceFactory_HPP

#include "umpire/resource/MemoryResourceFactory.hpp"

#include <cstddef>

namespace umpire {
namespace resource {

/*!
 * \brief Factory class for constructing MemoryResource objects that allocate
 * memory on specific NUMA nodes.
 */
class NumaMemoryResourceFactory :
  public MemoryResourceFactory
{
public:
  NumaMemoryResourceFactory(const int numa_node_);

  static std::size_t getNumberOfNumaNodes();

  bool isValidMemoryResourceFor(const std::string& name,
                                const MemoryResourceTraits traits = MemoryResourceTraits{}) noexcept;

  std::shared_ptr<MemoryResource> create(const std::string& name, int id);

private:
  const int numa_node;
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_NumaMemoryResourceFactory_HPP

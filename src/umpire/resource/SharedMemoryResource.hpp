//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_SharedMemoryResource_HPP
#define UMPIRE_SharedMemoryResource_HPP

#include "umpire/resource/MemoryResource.hpp"
#include "umpire/strategy/SharedAllocationStrategy.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

namespace umpire {
namespace resource {

class SharedMemoryResource :
  public MemoryResource,
  public strategy::SharedAllocationStrategy
{
  public:
    SharedMemoryResource(const std::string& name, int id, MemoryResourceTraits traits);
    virtual ~SharedMemoryResource() = default;

    MemoryResourceTraits getTraits() const noexcept override;

  protected:
    MemoryResourceTraits m_traits;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_SharedMemoryResource_HPP

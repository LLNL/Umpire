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
#ifndef UMPIRE_MemoryResource_HPP
#define UMPIRE_MemoryResource_HPP

#include "umpire/strategy/AllocationStrategy.hpp"

#include "umpire/util/MemoryResourceTraits.hpp"

namespace umpire {
namespace resource {

/*!
 * \brief Base class to represent the available hardware resources for memory
 * allocation in the system.
 *
 * Objects of this inherit from strategy::AllocationStrategy, allowing them to
 * be used directly.
 */
class MemoryResource :
  public strategy::AllocationStrategy
{
  public:
    /*!
     * \brief Construct a MemoryResource with the given name and id.
     *
     * \param name Name of the MemoryResource.
     * \param id ID of the MemoryResource (must be unique).
     *
     */
    MemoryResource(const std::string& name, int id, MemoryResourceTraits traits);

    virtual ~MemoryResource() = default;

    virtual void finalize() override
    {}

    MemoryResourceTraits getTraits();
  protected:
    MemoryResourceTraits m_traits;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_MemoryResource_HPP

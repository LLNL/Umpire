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
#ifndef UMPIRE_AllocationStrategyRegistry_HPP
#define UMPIRE_AllocationStrategyRegistry_HPP

#include <memory>
#include <list>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/AllocationStrategyFactory.hpp"

namespace umpire {

namespace strategy {

class AllocationStrategyRegistry {
  public:
    static AllocationStrategyRegistry& getInstance();

    std::shared_ptr<umpire::strategy::AllocationStrategy> makeAllocationStrategy(
      const std::string& name, 
      int id,
      const std::string& strategy,
      util::AllocatorTraits traits,
      std::vector<std::shared_ptr<AllocationStrategy> > providers);

    void registerAllocationStrategy(std::shared_ptr<AllocationStrategyFactory> factory);

  private:
    AllocationStrategyRegistry();

    AllocationStrategyRegistry(const AllocationStrategyRegistry&) = delete;

    AllocationStrategyRegistry& operator= (const AllocationStrategyRegistry&) = delete;

    static AllocationStrategyRegistry* s_allocator_registry_instance;

    std::list<std::shared_ptr<AllocationStrategyFactory> > m_allocator_factories;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_AllocationStrategyRegistry_HPP

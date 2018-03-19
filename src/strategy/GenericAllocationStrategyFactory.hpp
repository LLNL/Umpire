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
#ifndef UMPIRE_GenericAllocationStrategyFactory_HPP
#define UMPIRE_GenericAllocationStrategyFactory_HPP

#include "umpire/strategy/AllocationStrategyFactory.hpp"

namespace umpire {

namespace strategy {

template <typename ALLOC_STRATEGY>
class GenericAllocationStrategyFactory 
  : public AllocationStrategyFactory {
  public:
    GenericAllocationStrategyFactory(const std::string& name);

    bool isValidAllocationStrategyFor(const std::string& name);

    std::shared_ptr<AllocationStrategy> create(
        const std::string& name,
        int id,
        util::AllocatorTraits traits,
        std::vector<std::shared_ptr<AllocationStrategy> > providers);
  private:
    std::string m_name;
};

} // end of namespace strategy
} // end of namespace umpire

#include "umpire/strategy/GenericAllocationStrategyFactory.inl"

#endif // UMPIRE_GenericAllocationStrategyFactory_HPP

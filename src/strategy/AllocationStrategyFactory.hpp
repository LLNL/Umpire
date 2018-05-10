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
#ifndef UMPIRE_AllocationStrategyFactory_HPP
#define UMPIRE_AllocationStrategyFactory_HPP

#include <memory>
#include <string>
#include <vector>

#include "umpire/strategy/AllocationStrategy.hpp"

#include "umpire/util/AllocatorTraits.hpp"

namespace umpire {

namespace strategy {

class AllocationStrategyFactory {
  public:
    virtual bool isValidAllocationStrategyFor(const std::string& name) = 0;

    virtual std::shared_ptr<AllocationStrategy> create(
        const std::string& name,
        int id,
        util::AllocatorTraits traits,
        std::vector<std::shared_ptr<AllocationStrategy> > providers) = 0;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_AllocationStrategyFactory_HPP

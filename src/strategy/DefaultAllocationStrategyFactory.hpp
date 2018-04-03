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
#ifndef UMPIRE_DefaultAllocationStrategyFactory_HPP
#define UMPIRE_DefaultAllocationStrategyFactory_HPP

#include "umpire/strategy/AllocationStrategyFactory.hpp"

#include <memory>
#include <string>

namespace umpire {
namespace strategy {

class DefaultAllocationStrategyFactory :
  public AllocationStrategyFactory {
  public:
    bool isValidAllocationStrategyFor(const std::string& name);
    std::shared_ptr<AllocationStrategy> create();
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_DefaultAllocationStrategyFactory_HPP

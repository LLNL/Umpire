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
#include "umpire/strategy/DefaultAllocationStrategyFactory.hpp"

namespace umpire {
namespace resource {

DefaultAllocationStrategyFactory::DefaultAllocationStrategyFactory()

bool 
isValidAllocationStrategyFor(const std::string& name)
{
}

std::shared_ptr<AllocationStrategy> 
create();

} // end of namespace resource
} // end of namespace umpire

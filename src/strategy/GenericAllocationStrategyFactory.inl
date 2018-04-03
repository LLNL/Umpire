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
#ifndef UMPIRE_GenericAllocationStrategyFactory_INL
#define UMPIRE_GenericAllocationStrategyFactory_INL

#include "umpire/strategy/GenericAllocationStrategyFactory.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

template <typename ALLOC_STRATEGY>
GenericAllocationStrategyFactory<ALLOC_STRATEGY>::GenericAllocationStrategyFactory(const std::string& name):
  m_name(name)
{
}

template <typename ALLOC_STRATEGY>
bool 
GenericAllocationStrategyFactory<ALLOC_STRATEGY>::isValidAllocationStrategyFor(const std::string& name)
{
  return (name.compare(m_name) == 0);
}

template <typename ALLOC_STRATEGY>
std::shared_ptr<AllocationStrategy> 
GenericAllocationStrategyFactory<ALLOC_STRATEGY>::create(
    const std::string& name,
    int id,
    util::AllocatorTraits traits,
    std::vector<std::shared_ptr<AllocationStrategy> > providers)
{
  return std::make_shared<ALLOC_STRATEGY>(name, id, traits, providers);
}

}
}

#endif

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
#ifndef UMPIRE_Pool_HPP
#define UMPIRE_Pool_HPP

#include <memory>
#include <vector>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/AllocatorTraits.hpp"

namespace umpire {

namespace strategy {

class Pool :
  public AllocationStrategy
{
  public:
    Pool(
        const std::string& name,
        int id,
       util::AllocatorTraits traits,
       std::vector<std::shared_ptr<AllocationStrategy> > providers);

    void* allocate(size_t bytes);
    void deallocate(void* ptr);

    long getCurrentSize();
    long getHighWatermark();

    Platform getPlatform();

  private:
    void init();

    void** m_pointers;
    size_t* m_lengths;

    long m_current_size;
    long m_highwatermark;

    size_t m_slots;

    std::shared_ptr<umpire::strategy::AllocationStrategy> m_allocator;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_Pool_HPP

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
#ifndef UMPIRE_AllocationRecord_HPP
#define UMPIRE_AllocationRecord_HPP

#include <cstddef>
#include <memory>

namespace umpire {

namespace strategy {
  class AllocationStrategy;
}

namespace util {

struct AllocationRecord
{
  void* m_ptr;
  size_t m_size;
  const std::shared_ptr<strategy::AllocationStrategy>& m_strategy;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_AllocationRecord_HPP

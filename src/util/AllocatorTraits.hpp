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
#ifndef UMPIRE_AllocatorTraits_HPP
#define UMPIRE_AllocatorTraits_HPP

namespace umpire {
namespace util {

struct AllocatorTraits {
  size_t m_initial_size;
  size_t m_maximum_size;
  size_t m_number_allocations;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_AllocatorTraits_HPP

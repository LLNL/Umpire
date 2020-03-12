//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Allocator_INL
#define UMPIRE_Allocator_INL

#include "umpire/config.hpp"

#include "umpire/Allocator.hpp"

namespace umpire {

inline void*
Allocator::allocate(std::size_t bytes)
{

  umpire_ver_2_found = 0;

  return allocate_impl(bytes);
}

inline void
Allocator::deallocate(void* ptr)
{
  deallocate_impl(ptr);
}

} // end of namespace umpire

#endif // UMPIRE_Allocator_INL

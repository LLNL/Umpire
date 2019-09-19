//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Allocator_INL
#define UMPIRE_Allocator_INL

#include "umpire/config.hpp"

#include "umpire/Allocator.hpp"

namespace umpire {

void*
Allocator::allocate(std::size_t bytes)
{
  umpire_ver_1_found = 0;

  return allocate_impl(bytes);
}

void
Allocator::deallocate(void* ptr)
{
  deallocate_impl(ptr);
}

} // end of namespace umpire

#endif // UMPIRE_Allocator_INL

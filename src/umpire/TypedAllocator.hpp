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
#ifndef UMPIRE_TypedAllocator_HPP
#define UMPIRE_TypedAllocator_HPP

#include "umpire/Allocator.hpp"

namespace umpire {

template<typename T>
class TypedAllocator {
  public:
  typedef T value_type;

  TypedAllocator(Allocator allocator);

  T* allocate(size_t size);

  void deallocate(T* ptr, size_t size);

  private:
    umpire::Allocator m_allocator;
};

} // end of namespace umpire

#include "umpire/TypedAllocator.inl"

#endif // UMPIRE_TypedAllocator_HPP

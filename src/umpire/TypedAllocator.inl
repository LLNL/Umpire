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
#ifndef UMPIRE_TypedAllocator_INL
#define UMPIRE_TypedAllocator_INL

#include "umpire/util/Macros.hpp"

namespace umpire {

template<typename T>
TypedAllocator<T>::TypedAllocator(Allocator allocator) :
  m_allocator(allocator)
{
}

template<typename T>
template<typename U>
TypedAllocator<T>::TypedAllocator(const TypedAllocator<U>& other) :
  m_allocator(other.m_allocator)
{
}

template<typename T>
T* 
TypedAllocator<T>::allocate(size_t size)
{
  return static_cast<T*>(m_allocator.allocate(sizeof(T)*size));
}

template<typename T>
void 
TypedAllocator<T>::deallocate(T* ptr, size_t UMPIRE_UNUSED_ARG(size))
{
  m_allocator.deallocate(ptr);
}


} // end of namespace umpire

#endif // UMPIRE_TypedAllocator_INL

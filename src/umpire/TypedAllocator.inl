//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_TypedAllocator_INL
#define UMPIRE_TypedAllocator_INL

#include "umpire/util/Macros.hpp"

namespace umpire {

template <typename T>
TypedAllocator<T>::TypedAllocator(Allocator allocator) : m_allocator(allocator)
{
}

template <typename T>
template <typename U>
TypedAllocator<T>::TypedAllocator(const TypedAllocator<U>& other)
    : m_allocator(other.m_allocator)
{
}

template <typename T>
T* TypedAllocator<T>::allocate(std::size_t size)
{
  return static_cast<T*>(m_allocator.allocate(sizeof(T) * size));
}

template <typename T>
void TypedAllocator<T>::deallocate(T* ptr, std::size_t UMPIRE_UNUSED_ARG(size))
{
  m_allocator.deallocate(ptr);
}

} // end of namespace umpire

#endif // UMPIRE_TypedAllocator_INL

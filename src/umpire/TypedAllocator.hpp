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
#ifndef UMPIRE_TypedAllocator_HPP
#define UMPIRE_TypedAllocator_HPP

#include "umpire/Allocator.hpp"

namespace umpire {

/*!
 * \brief Allocator for objects of type T
 *
 * This class is an adaptor that allows using an Allocator to allocate objects
 * of type T. You can use this class as an allocator for STL containers like
 * std::vector.
 */
template<typename T>
class TypedAllocator {
  public:
  typedef T value_type;

  /*!
   *
   * \brief Construct a new TypedAllocator that will use allocator to allocate
   * data
   *
   * \param allocator Allocator to use for allocating memory.
   */
  TypedAllocator(Allocator allocator);

  /*
   * \brief Allocate size objects of type T.
   *
   * \param size The number of objects to allocate.
   *
   * \return Pointer to the start of the allocated memory.
   */
  T* allocate(size_t size);

  /*!
   * \brief Deallocate ptr, the passed size is ignored.
   * 
   * \param ptr Pointer to deallocate
   * \param size Size of allocation (ignored).
   */
  void deallocate(T* ptr, size_t size);

  private:
    umpire::Allocator m_allocator;
};

} // end of namespace umpire

#include "umpire/TypedAllocator.inl"

#endif // UMPIRE_TypedAllocator_HPP

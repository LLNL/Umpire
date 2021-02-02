//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_TypedAllocator_HPP
#define UMPIRE_TypedAllocator_HPP

#include "umpire/Allocator.hpp"


// forward declarations

namespace umpire {

template <typename T>
class TypedAllocator;

}

template <typename U, typename V>
bool operator==(const umpire::TypedAllocator<U>&, const umpire::TypedAllocator<V>&);

template <typename U, typename V>
bool operator!=(const umpire::TypedAllocator<U>&, const umpire::TypedAllocator<V>&);

namespace umpire {

/*!
 * \brief Allocator for objects of type T
 *
 * This class is an adaptor that allows using an Allocator to allocate objects
 * of type T. You can use this class as an allocator for STL containers like
 * std::vector.
 */
template <typename T>
class TypedAllocator {
 public:
  typedef T value_type;

  template <typename U>
  friend class TypedAllocator;

  /*!
   *
   * \brief Construct a new TypedAllocator that will use allocator to allocate
   * data
   *
   * \param allocator Allocator to use for allocating memory.
   */
  explicit TypedAllocator(Allocator allocator);

  template <typename U>
  TypedAllocator(const TypedAllocator<U>& other);

  /*
   * \brief Allocate size objects of type T.
   *
   * \param size The number of objects to allocate.
   *
   * \return Pointer to the start of the allocated memory.
   */
  T* allocate(std::size_t size);

  /*!
   * \brief Deallocate ptr, the passed size is ignored.
   *
   * \param ptr Pointer to deallocate
   * \param size Size of allocation (ignored).
   */
  void deallocate(T* ptr, std::size_t size);

  template <typename U, typename V>
  friend bool ::operator==(const TypedAllocator<U>&, const TypedAllocator<V>&);

  template <typename U, typename V>
  friend bool ::operator!=(const TypedAllocator<U>&, const TypedAllocator<V>&);

 private:
  umpire::Allocator m_allocator;
};

} // end of namespace umpire

#include "umpire/TypedAllocator.inl"

#endif // UMPIRE_TypedAllocator_HPP

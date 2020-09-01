//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_DeviceAllocator_HPP
#define UMPIRE_DeviceAllocator_HPP

#include "umpire/Allocator.hpp"

namespace umpire {

/*!
 * \brief Lightweight allocator for use in GPU code
 */
class DeviceAllocator {
 public:
  /*!
   *
   * \brief Construct a new DeviceAllocator that will use allocator to allocate
   * data
   *
   * \param allocator Allocator to use for allocating memory.
   */
  __host__ DeviceAllocator(Allocator allocator, size_t size);

  __host__ ~DeviceAllocator();

  __host__ __device__ DeviceAllocator(const DeviceAllocator& other);

  /*
   * \brief Allocate size objects of type T.
   *
   * \param size The number of objects to allocate.
   *
   * \return Pointer to the start of the allocated memory.
   */
  __device__ void* allocate(size_t size);

 private:
  umpire::Allocator m_allocator;

  char* m_ptr;
  unsigned int* m_counter;

  size_t m_size;

  bool m_child;
};

} // end of namespace umpire

#endif // UMPIRE_DeviceAllocator_HPP

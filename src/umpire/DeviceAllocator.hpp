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
  DeviceAllocator(Allocator allocator, size_t size);

  /*
   * \brief Allocate size objects of type T.
   *
   * \param size The number of objects to allocate.
   *
   * \return Pointer to the start of the allocated memory.
   */
  __device__
  void* allocate(size_t size);

  private:
    umpire::Allocator m_allocator;

    void* m_ptr;
    void* m_counter;

    size_t m_size;
};

} // end of namespace umpire

#include "umpire/DeviceAllocator.inl"

#endif // UMPIRE_DeviceAllocator_HPP

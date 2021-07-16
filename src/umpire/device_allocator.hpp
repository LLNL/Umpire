//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_DeviceAllocator_HPP
#define UMPIRE_DeviceAllocator_HPP

#include "umpire/allocator.hpp"
#include "umpire/resource.hpp"

namespace umpire {

/*!
 * \brief Lightweight allocator for use in GPU code
 */
template<typename T, typename Memory=strategy::allocation_strategy>
class device_allocator {
  public:
  /*!
   *
   * \brief Construct a new DeviceAllocator that will use allocator to allocate
   * data
   *
   * \param allocator Allocator to use for allocating memory.
   */
   __host__
  device_allocator(allocator<T> allocator, size_t size)
  {
    auto device = resource::device_memory::get();
    m_counter = static_cast<unsigned int*>(device->allocate(sizeof(unsigned int)));
    op::memset<platform::cuda>(m_counter, 0, 1);
  }

   __host__
  ~device_allocator()
  {
    if (!m_child) {
      auto device = resource::device_memory::get();
      device->deallocate(m_counter);

      m_allocator.deallocate(m_ptr);
    }
  }

  __host__ __device__
  device_allocator(const device_allocator& other) :
    m_allocator(other.m_allocator),
    m_ptr(other.m_ptr),
    m_counter(other.m_counter),
    m_size(other.m_size),
    m_child(true)
  {
  }

  /*
   * \brief Allocate size objects of type T.
   *
   * \param size The number of objects to allocate.
   *
   * \return Pointer to the start of the allocated memory.
   */
  __device__
  T* allocate(size_t size) {
    std::size_t counter = atomicAdd(m_counter, size);
    if (*m_counter > m_size) {
      //UMPIRE_ERROR("DeviceAllocator out of space");
    }
    return static_cast<T*>(m_ptr + counter*sizeof(T));
  }

  private:
    umpire::allocator<T, Memory> m_allocator;

    char* m_ptr;

    unsigned int* m_counter;

    size_t m_size;

    bool m_child;
};


} // end of namespace umpire

#endif // UMPIRE_DeviceAllocator_HPP

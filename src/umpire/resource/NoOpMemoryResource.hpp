//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_NoOpMemoryResource_HPP
#define UMPIRE_NoOpMemoryResource_HPP

#include "umpire/resource/MemoryResource.hpp"
#include "umpire/util/Platform.hpp"
#include "umpire/alloc/NoOpAllocator.hpp"

namespace umpire {
namespace resource {

/*!
 * \brief No-Op Memory allocator
 *
 * This NoOpMemoryResource has been created for benchmarking and
 * performance measurement purposes. This class will allocate a trivial amount
 * of memory in the constructor, and then increment pointers in the allocate
 * and deallocate functions. Thus, no malloc calls will be counted in the 
 * benchmarks and other function calls (etc) will be counted instead. The hope
 * is that more informative measurements and tracking can be done in the
 * benchmark than just focusing on the memory malloc calls.
 *
 */
class NoOpMemoryResource : public MemoryResource {
 public:
  /*!
   * \brief Construct a new NoOpMemoryResource and allocate a trivial 
   * amount of memory.
   */
  NoOpMemoryResource(Platform platform, const std::string& name, int id,
                     MemoryResourceTraits traits);

  /*!
   * \brief Resets allocation counter, frees initial memory.
   */
  ~NoOpMemoryResource();

  /*!
   * \brief Takes the pointer which was used in the constructor to allocate
   * trivial amount of memory and increments by n. Then increments n itself
   * so that each allocation has a unique index.
   *
   * \return void* 
   */
  void* allocate(std::size_t bytes);

  /*!
   * \brief Decrements the counter used by the allocate function.
   */
  void deallocate(void* ptr);

  std::size_t getCurrentSize() const noexcept;
  std::size_t getHighWatermark() const noexcept;

  bool isAccessibleFrom(Platform p) noexcept;
  
  Platform getPlatform() noexcept;

 protected:
  Platform m_platform;
  alloc::NoOpAllocator m_allocator;

 private:
  void* m_ptr;
  int m_count;
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_NoOpMemoryResource_HPP

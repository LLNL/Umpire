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

namespace umpire {
namespace resource {

/*!
 * \brief No-Op Memory allocator
 *
 * This NoOpMemoryResource has been created for benchmarking and
 * performance measurement purposes. This class increments pointers in the 
 * allocate function. Thus, no malloc calls will be counted in the 
 * benchmarks and other function calls (etc) will be counted instead. The hope
 * is that more informative measurements and tracking can be done in the
 * benchmark than just focusing on the memory malloc calls.
 *
 */
class NoOpMemoryResource : public MemoryResource {
 public:
  /*!
   * \brief Construct a new NoOpMemoryResource.
   */
  NoOpMemoryResource(Platform platform, const std::string& name, int id,
                     MemoryResourceTraits traits);

  /*!
   * \brief Default destructor.
   */
  ~NoOpMemoryResource();

  /*!
   * \brief Creates a pointer and increments it by bytes. 
   * Then, increments counter so that each allocation has a unique index.
   * NO MEMORY ALLOCATED!
   *
   * \return void* 
   */
  void* allocate(std::size_t bytes);

  /*!
   * \brief Does nothing.
   */
  void deallocate(void* ptr);

  std::size_t getCurrentSize() const noexcept;
  std::size_t getHighWatermark() const noexcept;

  bool isAccessibleFrom(Platform p) noexcept;
  
  Platform getPlatform() noexcept;

 protected:
  Platform m_platform;

 private:
  std::size_t m_count = (UINT64_C(1)<<48);
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_NoOpMemoryResource_HPP

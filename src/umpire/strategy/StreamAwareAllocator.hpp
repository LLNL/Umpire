//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_StreamAwareAllocator_HPP
#define UMPIRE_StreamAwareAllocator_HPP

#include "camp/camp.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {
namespace strategy {

/*!
 *
 * \brief Make an Allocator thread safe
 *
 * Using this AllocationStrategy will make the provided allocator thread-safe
 * by syncronizing access to the allocators interface.
 */
class StreamAwareAllocator : public AllocationStrategy {
 public:
  StreamAwareAllocator(const std::string& name, int id, Allocator allocator);

  void* allocate(void* stream, std::size_t bytes);
  void deallocate(void* stream, void* ptr, std::size_t size);

  Platform getPlatform() noexcept override;

  MemoryResourceTraits getTraits() const noexcept override;

private:
  void* allocate(std::size_t bytes) override;
  void deallocate(void* ptr, std::size_t size) override;

  std::vector<void*> m_registered_streams{0};
  std::vector<camp::resources::Event> m_registered_dealloc{0};

 protected:
  strategy::AllocationStrategy* m_allocator;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_StreamAwareAllocator_HPP

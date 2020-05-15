//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_SharedMemoryAllocator_HPP
#define UMPIRE_SharedMemoryAllocator_HPP

#include "umpire/strategy/SharedAllocationStrategy.hpp"

#include "umpire/util/Platform.hpp"

#include <iostream>
#include <string>

namespace umpire {

class ResourceManager;

class SharedMemoryAllocator {
  friend class ResourceManager;

  public:
    void* allocate(std::size_t bytes);
    void* allocate(std::string name, std::size_t bytes);
    void* get_allocation_by_name(std::string name);
    void deallocate(void* ptr);
    void release();

    std::size_t getSize(void* ptr) const;
    std::size_t getHighWatermark() const noexcept;
    std::size_t getCurrentSize() const noexcept;
    std::size_t getActualSize() const noexcept;
    std::size_t getAllocationCount() const noexcept;

    const std::string& getName() const noexcept;
    int getId() const noexcept;

    strategy::SharedAllocationStrategy* getAllocationStrategy() noexcept;

    Platform getPlatform() noexcept;
    SharedMemoryAllocator() = default;

    void set_foreman(int id);
    bool is_foreman();
    void synchronize();

    friend std::ostream& operator<<(std::ostream&, const SharedMemoryAllocator&);

  private:
    SharedMemoryAllocator(strategy::SharedAllocationStrategy* allocator) noexcept;

    strategy::SharedAllocationStrategy* m_allocator;
};

} // end of namespace umpire

#endif // UMPIRE_SharedMemoryAllocator_HPP

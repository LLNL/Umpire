//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_NamedSharedStrategy_HPP
#define UMPIRE_NamedSharedStrategy_HPP

#include "umpire/strategy/SharedAllocationStrategy.hpp"
#include "umpire/util/MemoryMap.hpp"

#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>

namespace umpire {

class SharedMemoryAllocator;

namespace strategy {

class NamedSharedStrategy :
  public SharedAllocationStrategy
{
  public:
    NamedSharedStrategy(const std::string& name, int id, SharedMemoryAllocator allocator) noexcept;
    ~NamedSharedStrategy();

    NamedSharedStrategy(const NamedSharedStrategy&) = delete;

    using SharedAllocationStrategy::allocate;
    void* allocate(std::string name, std::size_t bytes) override;
    void* get_allocation_by_name(std::string allocation_name) override;
    void deallocate(void* ptr) override;

    virtual void set_foreman(int id) override;
    virtual bool is_foreman() override;
    virtual void synchronize() override;

    Platform getPlatform() noexcept override;

  private:
    std::unordered_map<std::string, void*> m_name_to_pointer{};
    std::unordered_map<void*, std::string> m_pointer_to_name{};

    strategy::SharedAllocationStrategy* m_allocator;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_NamedSharedStrategy_HPP

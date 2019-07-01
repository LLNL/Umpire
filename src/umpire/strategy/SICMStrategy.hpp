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
#ifndef UMPIRE_SICMStrategy_HPP
#define UMPIRE_SICMStrategy_HPP

#include "umpire/strategy/AllocationStrategy.hpp"

#include "umpire/Allocator.hpp"
#include "umpire/util/SICM_device.hpp"

#include <sicm_low.h>

namespace umpire {

namespace strategy {

/*!
 * \brief Use SICM interface to locate memory to a specific SICM device.
 *
 * This AllocationStrategy provides a method of ensuring memory sits
 * on a specific SICM device. This can be used either for optimization,
 * or for moving memory between the host and devices.
 */
class SICMStrategy :
  public AllocationStrategy
{
  public:
    SICMStrategy(
        const std::string& name,
        int id,
        sicm_device_list devices,
        std::size_t max_size = 0);

    ~SICMStrategy();

    void* allocate(size_t bytes);
    void deallocate(void* ptr);

    std::size_t getCurrentSize() const noexcept;
    std::size_t getHighWatermark() const noexcept;

    Platform getPlatform() noexcept;

    sicm_arena getArena() const noexcept;

  private:
    std::size_t m_index;
    std::size_t m_max_size;
    sicm_arena m_arena;

    static sicm_device_list m_devices;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_SICMStrategy_HPP

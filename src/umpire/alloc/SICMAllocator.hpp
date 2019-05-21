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
#ifndef UMPIRE_SICMAllocator_HPP
#define UMPIRE_SICMAllocator_HPP

#include <cstdlib>
#include <list>
#include <map>
#include <set>
#include <vector>

#include "umpire/util/Macros.hpp"

extern "C"
{
#include <sicm_low.h>
}

namespace umpire {
namespace alloc {

/*!
 * \brief Uses SICM to allocate and deallocate memory into arenas.
 */
struct SICMAllocator
{
  SICMAllocator(const std::set <unsigned int> & devices)
    : devs(sicm_init()),
      allowed_devices(),
      best_device_index(0)
  {
    if (!devices.size()) {
      UMPIRE_ERROR("SICMAllocator construction failed due to lack of allowed devices");
    }

    // check if any of the devices are out of bounds
    for(const unsigned int dev : devices) {
      if (dev >= devs.count) {
        cleanup();
        UMPIRE_ERROR("SICMAllocator Bad device index: " << dev << " [0-" << devs.count << ")");
      }

      allowed_devices.push_back(dev);     // move the contents of devices into allowed devices
      UMPIRE_USE_VAR(arenas[dev]);        // create the map node
    }
  }

  ~SICMAllocator() {
    cleanup();
  }

  void cleanup() {
    {
      std::lock_guard <std::mutex> lock(arena_mutex);
      // only delete the arenas on devices that this instance is allowed to control
      for(const int dev : allowed_devices) {
        for(sicm_arena arena : arenas[dev]) {
          sicm_arena_destroy(arena);
        }
      }
    }
    sicm_fini();
  }

  /*!
   * \brief Allocate bytes of memory using sicm_alloc.
   *
   * \param bytes Number of bytes to allocate.
   * \return Pointer to start of the allocation.
   *
   * \throws umpire::util::Exception if memory cannot be allocated.
   */
  void* allocate(size_t bytes)
  {
    void* ret = nullptr;
    {
      std::lock_guard <std::mutex> lock(arena_mutex);

      // find best device
      const int best = allowed_devices[best_device_index];
      UMPIRE_LOG(Debug, "Best device to allocate on: " << best);
      best_device_index = (best_device_index + 1) % allowed_devices.size();

      // get list of arenas currently on the device
      std::list <sicm_arena> & arenas_on_device = arenas[best];

      // get an arena
      sicm_arena sa = nullptr;
      if (!arenas_on_device.size()) {
        UMPIRE_LOG(Debug, "Creating new arena on device " << best);
        if (!(sa = sicm_arena_create(0, &devs.devices[best]))) {
          cleanup();
          UMPIRE_ERROR("SICMAllocator Could not create arena on device " << best);
        }
      }
      else {
        UMPIRE_LOG(Debug, "Using existing arena on device " << best);

        // take first arena
        sa = arenas_on_device.front();

        // pop arena
        arenas_on_device.pop_front();
      }

      // allocate on arena
      UMPIRE_LOG(Debug, "Using arena located at " << sa << " on device " << best);
      ret = sicm_arena_alloc(sa, bytes);

      // push arena to back of list
      arenas_on_device.push_back(sa);
    }

    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);
    if  (ret == nullptr) {
      cleanup();
      UMPIRE_ERROR("SICM( bytes = " << bytes << " ) failed");
    } else {
      return ret;
    }
  }

  /*!
   * \brief Deallocate memory using sicm_free.
   *
   * \param ptr Address to deallocate.
   *
   * \throws umpire::util::Exception if memory cannot be free'd.
   */
  void deallocate(void* ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
    sicm_free(ptr);
  }

  sicm_device_list devs;
  std::vector <unsigned int> allowed_devices;
  std::size_t best_device_index; // this should be deleted once a better selector is created

  static std::mutex arena_mutex;
  static std::map <unsigned int, std::list <sicm_arena> > arenas; // device index -> arenas on that device
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_SICMAllocator_HPP

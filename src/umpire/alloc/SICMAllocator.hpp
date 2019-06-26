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
#include <sched.h>
#include <vector>

#include "umpire/util/Macros.hpp"
#include "umpire/util/SICM_device.hpp"

#include <sicm_low.h>

namespace umpire {
namespace alloc {

/*!
 * \brief Uses SICM to allocate and deallocate memory into arenas.
 */
struct SICMAllocator
{
  SICMAllocator(const std::string& name, const std::vector <unsigned int>& devices, const sicm::device_chooser_t& chooser)
    : devs(sicm_init()),
      name(name),
      allowed_devices(devices),
      chooser(chooser)
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

      std::lock_guard <std::mutex> lock(arena_mutex);
      if (arenas.find(dev) != arenas.end()) {
        UMPIRE_ERROR("SICMAllocator Device has already been used in aother SICMAllocator: " << dev);
      }
    }

    // check if the device chooser is NULL
    if (!chooser) {
      UMPIRE_ERROR("SICMAllocator Bad device chooser.");
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
      // find best device
      const unsigned int best = chooser(sched_getcpu(), bytes, allowed_devices, devs);
      UMPIRE_LOG(Debug, "Best " << name << " device to allocate on: " << best);

      std::lock_guard <std::mutex> lock(arena_mutex);

      // get list of arenas currently on the device
      // create new mapping if best isn't found
      std::list<sicm_arena>& arenas_on_device = arenas[best];

      // get an arena
      sicm_arena sa = nullptr;
      if (!arenas_on_device.size()) {
        UMPIRE_LOG(Debug, "Creating new " << name << " arena on device " << best);
        if (!(sa = sicm_arena_create(0, &devs.devices[best]))) {
          cleanup();
          UMPIRE_ERROR("SICMAllocator Could not create " << name << " arena on device " << best);
        }
      }
      else {
        UMPIRE_LOG(Debug, "Using existing " << name << " arena on device " << best);

        // take first arena
        sa = arenas_on_device.front();

        // pop arena
        arenas_on_device.pop_front();
      }

      // allocate on arena
      UMPIRE_LOG(Debug, "Using " << name << " arena located at " << sa << " on device " << best);
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

  const sicm_device_list devs;
  const std::string name;
  std::vector <unsigned int> allowed_devices;
  const sicm::device_chooser_t& chooser;

  static std::mutex arena_mutex;
  static std::map <unsigned int, std::list <sicm_arena> > arenas; // device index -> arenas on that device
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_SICMAllocator_HPP

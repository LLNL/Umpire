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
#include "umpire/util/SICM_device.hpp"
#include "umpire/util/Macros.hpp"

#include <vector>

std::ostream& operator<<(std::ostream& stream, const sicm_device& device) {
  static const std::string units[] = {"K", "M", "G", "T"};

  std::size_t unit = 0;
  std::size_t page_size = device.page_size;

  while (page_size >= 1024) {
    page_size >>= 10;
    unit++;
  }

  return stream << "{"
                << sicm_device_tag_str(device.tag) << ", "
                << device.node << ", "
                << page_size << units[unit] << "B"
                << "}";
}

std::ostream& operator<<(std::ostream& stream, const sicm_device_list& device_list) {
  stream << device_list.count << " SICM devices:";
  for(unsigned int i = 0; i < device_list.count; i++) {
    stream << " " << *device_list.devices[i];
  }

  return stream;
}

namespace umpire {
namespace sicm {

std::shared_ptr<sicm_device_list> get_devices(const struct sicm_device_list& devs, const umpire::Platform& platform, int page_size) {
  std::vector<unsigned int> indicies;
  page_size >>= 10; // page_size in SICM is in units of 1K
  switch (platform) {
    case umpire::Platform::cpu:
      for(unsigned int i = 0; i < devs.count; i++) {
        if ((devs.devices[i]->tag == SICM_DRAM) &&
            (devs.devices[i]->page_size == page_size)) {
          indicies.push_back(i);
        }
        else if ((devs.devices[i]->tag == SICM_KNL_HBM) &&
                 (devs.devices[i]->page_size == page_size)) {
          indicies.push_back(i);
        }
      }
      break;
#if defined(UMPIRE_ENABLE_CUDA)
    case umpire::Platform::cuda:
      for(unsigned int i = 0; i < devs.count; i++) {
        if ((devs.devices[i]->tag == SICM_POWERPC_HBM) &&
            (devs.devices[i]->page_size == page_size)) {
          indicies.push_back(i);
        }
      }
      break;
#endif
    default:
        break;
  }

  // copy pointers into this device list
  std::shared_ptr<sicm_device_list> found(new sicm_device_list, [](sicm_device_list* ptr){ sicm_device_list_free(ptr); delete ptr; });
  found->count = indicies.size();
  found->devices = static_cast<sicm_device**>(calloc(indicies.size(), sizeof(sicm_device*)));
  for(decltype(indicies)::size_type i = 0; i < indicies.size(); i++) {
    found->devices[i] = devs.devices[indicies[i]];
  }

  return found;
}

} // end namespace sicm
} // end namespace umpire

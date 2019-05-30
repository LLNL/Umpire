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
#ifndef UMPIRE_sicm_HPP
#define UMPIRE_sicm_HPP

#include "umpire/util/Platform.hpp"

#include <vector>

extern "C" {
#include <sicm_low.h>
}

namespace umpire {
namespace sicm {

// gets set of devices suitable for a given platform
// returns device indicies, not numa nodes
std::vector<unsigned int> get_devices(const struct sicm_device_list& devs, const umpire::Platform& platform);

// get the device index that works best when running on a given CPU
unsigned int best_device(const int running_at,
                         const std::size_t size,
                         const std::vector <unsigned int>& allowed_devices,
                         const sicm_device_list& devs);

} // end namespace sicm
} // end namespace umpire


#endif // UMPIRE_sicm_HPP

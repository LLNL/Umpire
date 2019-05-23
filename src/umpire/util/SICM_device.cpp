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

namespace umpire {
namespace sicm {

unsigned int best_device(const int UMPIRE_UNUSED_ARG(running_at),
                             const std::size_t UMPIRE_UNUSED_ARG(size),
                             const std::vector <unsigned int> & allowed_devices,
                             const sicm_device_list & UMPIRE_UNUSED_ARG(devs)) {
    static std::size_t index = 0;
    const unsigned int dev = allowed_devices[index % allowed_devices.size()];
    index = (index + 1) % allowed_devices.size();
    return dev;
}

} // end namespace sicm
} // end namespace umpire

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

std::vector<unsigned int> get_devices(const struct sicm_device_list& devs, const umpire::Platform& platform) {
    std::vector<unsigned int> devices;
    switch (platform) {
        case umpire::Platform::cpu:
            for(unsigned int i = 0; i < devs.count; i++) {
                switch (devs.devices[i].tag) {
                    case SICM_DRAM:
                    case SICM_KNL_HBM:
                        devices.push_back(i);
                        break;
                    default:
                        break;
                }
            }
            break;
#if defined(UMPIRE_ENABLE_CUDA)
        case umpire::Platform::cuda:
            for(unsigned int i = 0; i < devs.count; i++) {
                if (devs.devices[i].tag == SICM_POWERPC_HBM) {
                    devices.push_back(i);
                }
            }
            break;
#endif
        default:
            break;
    }

    return devices;
}

unsigned int best_device(const int UMPIRE_UNUSED_ARG(running_at),
                         const std::size_t UMPIRE_UNUSED_ARG(size),
                         const std::vector <unsigned int>& allowed_devices,
                         const sicm_device_list& UMPIRE_UNUSED_ARG(devs)) {
    static std::size_t index = 0;
    const unsigned int dev = allowed_devices[index % allowed_devices.size()];
    index = (index + 1) % allowed_devices.size();
    return dev;
}

} // end namespace sicm
} // end namespace umpire

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

#include <memory>

#include <sicm_low.h>

std::ostream& operator<<(std::ostream& stream, const sicm_device_list& device_list);

namespace umpire {
namespace sicm {

// gets set of devices suitable for a given platform
// returns device indicies, not numa nodes
std::shared_ptr<sicm_device_list> get_devices(const struct sicm_device_list& devs, const umpire::Platform& platform, int page_size);

} // end namespace sicm
} // end namespace umpire


#endif // UMPIRE_sicm_HPP

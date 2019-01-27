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
#ifndef UMPIRE_DetectVendor_HPP
#define UMPIRE_DetectVendor_HPP

#include "umpire/resource/MemoryResourceTraits.hpp"

namespace umpire {

resource::MemoryResourceTraits::vendor_type CpuVendorType() noexcept;

}

#endif // UMPIRE_DetectVendor_HPP

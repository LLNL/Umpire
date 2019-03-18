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
#ifndef UMPIRE_detect_vendor_HPP
#define UMPIRE_detect_vendor_HPP

#include "umpire/util/MemoryResourceTraits.hpp"

namespace umpire {

MemoryResourceTraits::vendor_type cpu_vendor_type() noexcept;

} // end namespace umpire

#endif // UMPIRE_detect_vendor_HPP

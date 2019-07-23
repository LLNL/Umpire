//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_detect_vendor_HPP
#define UMPIRE_detect_vendor_HPP

#include "umpire/util/MemoryResourceTraits.hpp"

namespace umpire {

MemoryResourceTraits::vendor_type cpu_vendor_type() noexcept;

} // end namespace umpire

#endif // UMPIRE_detect_vendor_HPP

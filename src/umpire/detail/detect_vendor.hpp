//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "umpire/MemoryResourceTraits.hpp"

namespace umpire {
namespace detail {

constexpr MemoryResourceTraits::vendor_type cpu_vendor_type() noexcept;

}
} // end namespace umpire

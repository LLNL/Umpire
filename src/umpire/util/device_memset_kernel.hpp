////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_device_memset_kernel_HPP
#define UMPIRE_device_memset_kernel_HPP

#include <string.h>

#include "umpire/util/error.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {

/*!
 * \brief Launch a device kernel to set DEVICE memory to a value.
 *
 * \tparam T Type for the value used in the memset.
 * \param alloc Umpire Allocator with the memory to memset.
 * \param n Number of elements.
 * \param value Value to assign to each element (may be 0, -1, NaN, etc.).
 */
template <typename T>
inline void device_memset(Umpire::Allocator, std::size_t n, T value)

} // end of namespace umpire

#endif // UMPIRE_device_memset_kernel_HPP

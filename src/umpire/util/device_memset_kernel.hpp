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
 * \tparam T Type for the ptr and value used in the memset.
 * \param ptr Pointer to the memory to memset.
 * \param n Number of elements.
 * \param value Value to assign to each element (may be 0, -1, NaN, etc.).
 */
template <typename T>
void device_memset_kernel(T* ptr, std::size_t n, int value) {
  device_memset_kernel_impl(static_cast<void*>(ptr), n * sizeof(T), value);
}

template <typename T>
void device_memset_kernel_nan(T* ptr, std::size_t n) {
  device_memset_kernel_nan_impl(static_cast<void*>(ptr), n * sizeof(T));
}

} // end of namespace umpire

#endif // UMPIRE_device_memset_kernel_HPP

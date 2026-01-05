//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_find_first_set_HPP
#define UMPIRE_find_first_set_HPP

namespace umpire {
namespace util {

/*!
 * \brief Find the first (least significant) bit set in \p i.
 *
 * \param i Bits.
 *
 * \return Index of the first bit set. Bits are numbered starting at 1, the
 * least significant bit. A return value of 0 means that no bits were set.
 */
inline int find_first_set(int i)
{
#if defined(_MSC_VER)
  unsigned long index;
  unsigned long i_l = static_cast<unsigned long>(i);
  int bit = 0;
  if (_BitScanForward(&index, i_l)) {
    bit = static_cast<int>(index) + 1;
  }
  return bit;
#else
  return ffs(i);
#endif
}

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_find_first_set_HPP

//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Platform_HPP
#define UMPIRE_Platform_HPP

namespace umpire {

enum class Platform {
  none,
  cpu,
  cuda,
  rocm,
  hip,
  omp
};

} // end of namespace umpire

#endif

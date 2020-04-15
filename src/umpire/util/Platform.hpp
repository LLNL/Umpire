//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Platform_HPP
#define UMPIRE_Platform_HPP

// #include "camp/resource/platform.hpp"

namespace umpire {

// using Platform = camp::resources::Platform;

enum class Platform {
    undefined = 0,
    host = 1,
    cuda = 2,
    omp_target = 4,
    mpi_shmem = 5,
    hip = 8
};

} // end of namespace umpire

#endif

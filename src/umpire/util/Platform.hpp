//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Platform_HPP
#define UMPIRE_Platform_HPP

namespace umpire {

    enum class Platform {
        undefined = 0,
        host = 1,
        cuda = 2,
        omp_target = 4,
        hip = 8,
        mpi_shmem = 9,
        sycl = 16
    };

} // end of namespace umpire

#endif

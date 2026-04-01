//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_runtime_error_HPP
#define UMPIRE_runtime_error_HPP

#include <string>

#include "fmt/format.h"
#include "../../../include/umpire/error.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {

using out_of_memory_error = out_of_memory;
using unknown_pointer_error = unknown_allocation;

class resource_error : public umpire::runtime_error {
 public:
  using runtime_error::runtime_error;
};

} // end of namespace umpire

#if defined(__CUDA_ARCH__)
#define UMPIRE_ERROR(type, msg) asm("trap;");
#elif defined(__HIP_DEVICE_COMPILE__)
#define UMPIRE_ERROR(type, msg) abort();
#else
#define UMPIRE_ERROR(type, msg)                                           \
  {                                                                       \
    type e{msg, std::string{__FILE__}, __LINE__};                         \
    UMPIRE_LOG(Error, e.what());                                          \
    umpire::util::flush_files();                                          \
    throw e;                                                              \
  }
#endif

#endif // UMPIRE_runtime_error_HPP

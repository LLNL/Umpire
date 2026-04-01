//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_resource_null_resource_HPP
#define UMPIRE_resource_null_resource_HPP

#include <cstddef>
#include <string>

#include "umpire/memory_resource.hpp"
#include "umpire/platform.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace resource {

enum class null_behavior {
  return_nullptr,
  throw_exception
};

struct null_allocator {
  char* allocate(std::size_t)
  {
    return nullptr;
  }

  void deallocate(char*, std::size_t)
  {
  }
};

template<null_behavior Behavior = null_behavior::throw_exception>
class null_resource : public memory_resource<undefined_platform, null_allocator, false> {
private:
  using base = memory_resource<undefined_platform, null_allocator, false>;

public:
  explicit null_resource(const std::string& name = "NULL")
    : base(name)
  {
  }

  void* allocate(std::size_t size) override
  {
    if constexpr (Behavior == null_behavior::return_nullptr) {
      (void)size;
      return nullptr;
    } else {
      UMPIRE_ERROR(out_of_memory_error,
                   "null_resource: intentional allocation failure");
    }
  }

  void deallocate(void* ptr) override
  {
    (void)ptr;
  }
};

using default_null_resource = null_resource<null_behavior::throw_exception>;
using silent_null_resource = null_resource<null_behavior::return_nullptr>;

} // namespace resource
} // namespace umpire

#endif // UMPIRE_resource_null_resource_HPP

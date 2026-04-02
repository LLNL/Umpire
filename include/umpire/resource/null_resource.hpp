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

//! \brief Controls whether `null_resource` returns `nullptr` or throws.
enum class null_behavior {
  return_nullptr,
  throw_exception
};

/*!
 * \brief Allocator wrapper that never produces storage.
 *
 * `null_resource` uses this allocator to model deliberate allocation failure
 * while still satisfying the `memory_resource` allocator interface.
 */
struct null_allocator {
  //! \brief Always return `nullptr`.
  char* allocate(std::size_t)
  {
    return nullptr;
  }

  //! \brief Ignore deallocation requests.
  void deallocate(char*, std::size_t)
  {
  }
};

/*!
 * \brief Resource that intentionally never provides usable memory.
 *
 * This resource is useful for tests, sentinel configurations, and code paths
 * that need an API v2 memory object representing guaranteed failure.
 *
 * \tparam Behavior Selects whether allocation returns `nullptr` or throws.
 */
template<null_behavior Behavior = null_behavior::throw_exception>
class null_resource : public memory_resource<undefined_platform, null_allocator, false> {
private:
  using base = memory_resource<undefined_platform, null_allocator, false>;

public:
  /*!
   * \brief Construct a null resource.
   *
   * \param name Name exposed through the registry and diagnostics.
   */
  explicit null_resource(const std::string& name = "NULL")
    : base(name)
  {
  }

  /*!
   * \brief Reject an allocation request according to `Behavior`.
   *
   * \param size Requested byte count.
   * \return `nullptr` when `Behavior` is `return_nullptr`.
   *
   * \throws out_of_memory_error when `Behavior` is `throw_exception`.
   */
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

  /*!
   * \brief Ignore deallocation requests.
   *
   * \param ptr Pointer value to ignore.
   */
  void deallocate(void* ptr) override
  {
    (void)ptr;
  }
};

//! \brief Null resource variant that throws on every allocation.
using default_null_resource = null_resource<null_behavior::throw_exception>;
//! \brief Null resource variant that silently returns `nullptr`.
using silent_null_resource = null_resource<null_behavior::return_nullptr>;

} // namespace resource
} // namespace umpire

#endif // UMPIRE_resource_null_resource_HPP

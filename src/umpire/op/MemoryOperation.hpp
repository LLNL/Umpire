//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MemoryOperation_HPP
#define UMPIRE_MemoryOperation_HPP

#include <cstddef>

#include "camp/resource.hpp"
#include "umpire/util/AllocationRecord.hpp"

namespace umpire {
namespace op {

/*!
 * \brief Base class of an operation on memory.
 *
 * Neither the transfrom or apply methods are pure virtual, so inheriting
 * classes only need overload the appropriate method. However, both methods
 * will throw an error if called.
 */
class MemoryOperation {
 public:
  virtual ~MemoryOperation() = default;

  /*!
   * \brief Transfrom length bytes of memory from src_ptr to dst_ptr.
   *
   * \param src_ptr Pointer to source memory location.
   * \param dst_ptr Pointer to destinatino memory location.
   * \param src_allocation AllocationRecord of source.
   * \param dst_allocation AllocationRecord of destination.
   * \param length Number of bytes to transform.
   *
   * \throws util::Exception
   */
  virtual void transform(void* src_ptr, void** dst_ptr,
                         util::AllocationRecord* src_allocation,
                         util::AllocationRecord* dst_allocation,
                         std::size_t length);

  virtual camp::resources::Event transform_async(
      void* src_ptr, void** dst_ptr, util::AllocationRecord* src_allocation,
      util::AllocationRecord* dst_allocation, std::size_t length,
      camp::resources::Resource& ctx);

  /*!
   * \brief Apply val to the first length bytes of src_ptr.
   *
   * \param src_ptr Pointer to source memory location.
   * \param src_allocation AllocationRecord of source.
   * \param val Value to apply.
   * \param length Number of bytes to modify.
   *
   * \throws util::Exception
   */
  virtual void apply(void* src_ptr, util::AllocationRecord* src_allocation,
                     int val, std::size_t length);

  virtual camp::resources::Event apply_async(
      void* src_ptr, util::AllocationRecord* src_allocation, int val,
      std::size_t length, camp::resources::Resource& ctx);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_MemoryOperation_HPP

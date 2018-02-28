#ifndef UMPIRE_MemoryOperation_HPP
#define UMPIRE_MemoryOperation_HPP

#include <cstddef>

#include "umpire/util/AllocationRecord.hpp"

namespace umpire {
namespace op {

/*!
 * \brief Base class of an operation on memory.
 *
 * Neither the transfrom or apply methods are pure virtual, so inheriting
 * classes only need overload the appropriate method.
 */
class MemoryOperation {
  public:
    /*!
     * \brief Transfrom length bytes of memory from src_ptr to dst_ptr.
     *
     * \param src_ptr Pointer to source memory location.
     * \param dst_ptr Pointer to destinatino memory location.
     * \param src_allocation AllocationRecord of source.
     * \param dst_allocation AllocationRecord of destination.
     * \param length Number of bytes to transform.
     */
    virtual void transform(
        void* src_ptr,
        void* dst_ptr,
        util::AllocationRecord *src_allocation,
        util::AllocationRecord *dst_allocation,
        size_t length);

    /*!
     * \brief Apply val to the first length bytes of src_ptr.
     *
     * \param src_ptr Pointer to source memory location.
     * \param src_allocation AllocationRecord of source.
     * \param val Value to apply.
     * \param length Number of bytes to modify.
     */
    virtual void apply(
        void* src_ptr,
        util::AllocationRecord *src_allocation,
        int val,
        size_t length);
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_MemoryOperation_HPP

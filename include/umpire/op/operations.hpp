#pragma once

#include <cstdlib>
#include <stdexcept>

#include "camp/resource.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/config.hpp"

namespace umpire {
namespace op {

/*!
 * \brief Base metadata for a typed memory operation.
 *
 * Operation tags are used with the dispatch layer to select platform-specific
 * implementations at compile time and runtime.
 */
struct operation {
  //! \brief Number of platform type parameters required by the operation.
  static constexpr int arity = -1;
  //! \brief Human-readable operation name used for diagnostics.
  static constexpr const char* name = "UNKNOWN";
};

/*!
 * \brief Operation tag for memory copies between two platforms.
 *
 * \tparam Src Source platform tag.
 * \tparam Dst Destination platform tag.
 */
template <typename Src, typename Dst>
struct copy : public operation {
  static constexpr int arity = 2;
  static constexpr const char* name = "COPY";
};

/*!
 * \brief Operation tag for filling memory on a single platform.
 *
 * \tparam Src Platform tag describing the destination memory.
 */
template <typename Src>
struct memset : public operation {
  static constexpr int arity = 1;
  static constexpr const char* name = "MEMSET";
};

/*!
 * \brief Operation tag for reallocation on a single platform.
 *
 * The generic reallocate path performs allocate-copy-free semantics using the
 * owner discovered from allocation tracking.
 *
 * \tparam Src Platform tag describing the underlying allocation.
 */
template <typename Src>
struct reallocate : public operation {
  static constexpr int arity = 1;
  static constexpr const char* name = "REALLOCATE";

  /*!
   * \brief Reallocate a typed pointer synchronously.
   *
   * \tparam T Pointee type. For non-void pointers, `new_size` is an element count.
   * \param ptr Address of the pointer to reallocate.
   * \param new_size Requested element count or byte count for `void`.
   * \return Updated pointer value.
   */
  template <typename T>
  static T* exec(T** ptr, std::size_t new_size);

  /*!
   * \brief Reallocate a typed pointer asynchronously when the backend supports it.
   *
   * \tparam T Pointee type. For non-void pointers, `new_size` is an element count.
   * \param ptr_ptr Address of the pointer to reallocate.
   * \param new_size Requested element count or byte count for `void`.
   * \param ctx Execution resource describing the asynchronous context.
   * \return Event proxy tracking completion of the reallocation sequence.
   */
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(T** ptr_ptr, std::size_t new_size,
                                                                     camp::resources::Resource& ctx);

  /*!
   * \brief Reallocate a raw `void*` synchronously using byte semantics.
   *
   * \param ptr_ptr Address of the pointer to reallocate.
   * \param new_size Requested size in bytes.
   * \return Updated pointer value.
   */
  static void* exec(void** ptr_ptr, std::size_t new_size);

  /*!
   * \brief Reallocate a raw `void*` asynchronously using byte semantics.
   *
   * \param ptr_ptr Address of the pointer to reallocate.
   * \param new_size Requested size in bytes.
   * \param ctx Execution resource describing the asynchronous context.
   * \return Event proxy tracking completion of the reallocation sequence.
   */
  static camp::resources::EventProxy<camp::resources::Resource> exec(void** ptr_ptr, std::size_t new_size,
                                                                     camp::resources::Resource& ctx);
};

#define DEFINE_ADVICE_OP(op_name, name_str)                                                        \
  template <typename Src>                                                                          \
  struct op_name : public operation {                                                              \
    static constexpr int arity = 1;                                                                \
    static constexpr const char* name = #op_name;                                                  \
                                                                                                   \
    static void exec(void*, int, std::size_t)                                                      \
    {                                                                                              \
      throw std::runtime_error("Memory advice not supported for this platform");                   \
    }                                                                                              \
                                                                                                   \
    static camp::resources::EventProxy<camp::resources::Resource> exec(void*, int, std::size_t,    \
                                                                       camp::resources::Resource&) \
    {                                                                                              \
      throw std::runtime_error("Memory advice not supported for this platform");                   \
    }                                                                                              \
  };

DEFINE_ADVICE_OP(set_accessed_by, "SET_ACCESSED_BY")
DEFINE_ADVICE_OP(unset_accessed_by, "UNSET_ACCESSED_BY")
DEFINE_ADVICE_OP(set_preferred_location, "SET_PREFERRED_LOCATION")
DEFINE_ADVICE_OP(unset_preferred_location, "UNSET_PREFERRED_LOCATION")
DEFINE_ADVICE_OP(set_read_mostly, "SET_READ_MOSTLY")
DEFINE_ADVICE_OP(unset_read_mostly, "UNSET_READ_MOSTLY")
DEFINE_ADVICE_OP(set_coarse_grain, "SET_COARSE_GRAIN")
DEFINE_ADVICE_OP(unset_coarse_grain, "UNSET_COARSE_GRAIN")
DEFINE_ADVICE_OP(prefetch, "PREFETCH")

#undef DEFINE_ADVICE_OP

} // namespace op
} // namespace umpire

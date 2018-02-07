#ifndef UMPIRE_AllocationStrategy_HPP
#define UMPIRE_AllocationStrategy_HPP

#include "umpire/util/Platform.hpp"

#include <memory>
#include <cstddef>

namespace umpire {
namespace strategy {

/*!
 * \brief Allocator provides a unified interface to all Umpire classes that can
 * be used to allocate and free data.
 */
class AllocationStrategy :
  public std::enable_shared_from_this<AllocationStrategy>
{
  public:
    AllocationStrategy(const std::string& name, int id);

    /*!
     * \brief Allocate bytes of memory.
     *
     * \param bytes Number of bytes to allocate.
     *
     * \return Pointer to start of allocation.
     */
    virtual void* allocate(size_t bytes) = 0;

    /*!
     * \brief Free the memory at ptr.
     *
     * \param ptr Pointer to free.
     */
    virtual void deallocate(void* ptr) = 0;

    virtual long getCurrentSize() = 0;
    virtual long getHighWatermark() = 0;

    virtual Platform getPlatform()  = 0;

    std::string getName();

    int getId();

  protected:
    std::string m_name;

    int m_id;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_AllocationStrategy_HPP

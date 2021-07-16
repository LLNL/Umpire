//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "umpire/strategy/allocation_strategy.hpp"

#include <cstddef>
#include <cstring>
#include <vector>
#if !defined(_MSC_VER)
#define  _XOPEN_SOURCE_EXTENDED 1
#include <strings.h>
#endif

namespace umpire {
namespace strategy {

namespace {

inline int fixed_pool_ffs(int i)
{
#if defined(_MSC_VER)
  unsigned long bit;
  unsigned long i_l = static_cast<unsigned long>(i);
  _BitScanForward(&bit, i_l);
  return static_cast<int>(bit);
#else
  return ffs(i);
#endif
}

static constexpr std::size_t bits_per_int = sizeof(int) * 8;
}

/*!
 * \brief Pool for fixed size allocations
 *
 * This AllocationStrategy provides an efficient pool for fixed size
 * allocations, and used to quickly allocate and deallocate objects.
 */
template<typename Memory=memory, bool Tracking=true>
class fixed_pool :
  public allocation_strategy
{

  private:
    struct Pool {
      Memory* strategy;
      char* data;
      int* avail;
      std::size_t num_avail;

      Pool(Memory* allocation_strategy,
           const std::size_t object_bytes, 
           const std::size_t objects_per_pool,
           const std::size_t avail_bytes) :
      strategy(allocation_strategy),
      data(reinterpret_cast<char*>(strategy->allocate(object_bytes * objects_per_pool))),
      avail(reinterpret_cast<int*>(std::malloc(avail_bytes))),
      num_avail(objects_per_pool)  
      { 
        // Set all bits to 1
        const unsigned char not_zero = static_cast<unsigned char>(~0);
        std::memset(avail, not_zero, avail_bytes);
      }
    };

    void new_pool() {
      pools_.emplace_back(memory_, object_bytes_, objects_per_pool_, bytes_available_ * sizeof(int));
      bytes_actual_ += bytes_available_ + bytes_data_;
    }

    void *allocInPool(Pool &p)
    {
      if (!p.num_avail)
        return nullptr;

      for (unsigned int int_index = 0; int_index < bytes_available_; ++int_index)
      {
        // Return the index of the first 1 bit
        const int bit_index = fixed_pool_ffs(p.avail[int_index]) - 1;
        if (bit_index >= 0)
        {
          const std::size_t index = int_index * bits_per_int + bit_index;
          if (index < objects_per_pool_)
          {
            // Flip bit 1 -> 0
            p.avail[int_index] ^= 1 << bit_index;
            p.num_avail--;
            return static_cast<void *>(p.data + object_bytes_ * index);
          }
        }
      }
      UMPIRE_ASSERT("FixedPool::allocInPool(): num_avail > 0, but no available slots" && 0);
      return nullptr;
    }

  public:
    using platform = typename Memory::platform;
    /*!
     * \brief Constructs a FixedPool.
     *
     * \param name The allocator name for reference later in ResourceManager
     * \param id The allocator id for reference later in ResourceManager
     * \param allocator Used for data allocation. It uses std::malloc
     * for internal tracking of these allocations.
     * \param object_bytes The fixed size (in bytes) for each allocation
     * \param objects_per_pool Number of objects in each sub-pool
     * internally. Performance likely improves if this is large, at
     * the cost of memory usage. This does not have to be a multiple
     * of sizeof(int)*8, but it will also likely improve performance
     * if so.
     */
    fixed_pool(const std::string& name,
               Memory* memory,
               const std::size_t object_bytes,
               const std::size_t objects_per_pool = 64 * sizeof(int) * 8) noexcept :
      allocation_strategy{name},
      memory_{memory},
      object_bytes_{object_bytes},
      objects_per_pool_{objects_per_pool},
      bytes_data_{object_bytes_ * objects_per_pool_},
      bytes_available_{objects_per_pool/bits_per_int + 1},
      pools_{}
    {
      new_pool();
    }

    ~fixed_pool()
    {
      std::vector<void *> leaked_addrs{};

      for (auto &p : pools_) {
        if (objects_per_pool_ != p.num_avail) {
          for (unsigned int int_index = 0; int_index < bytes_available_; ++int_index)
            for (unsigned int bit_index = 0; bit_index < bits_per_int; ++bit_index) {
              if (!(p.avail[int_index] & 1 << bit_index)) {
                const std::size_t index{int_index * bits_per_int + bit_index};
                leaked_addrs.push_back(
                    static_cast<void *>(p.data + object_bytes_ * index));
              }
            }
        }
      }

      if (leaked_addrs.size() > 0) {
        const std::size_t max_addr{25};
        std::stringstream ss;
        ss << "There are " << leaked_addrs.size() << " addresses";
        ss << " not deallocated at destruction. This will cause leak(s). ";
        if (leaked_addrs.size() <= max_addr)
          ss << "Addresses:";
        else
          ss << "First " << max_addr << " addresses:";
        for (std::size_t i = 0; i < std::min(max_addr, leaked_addrs.size()); ++i) {
          if (i % 5 == 0)
            ss << "\n\t";
          ss << " " << leaked_addrs[i];
        }
        UMPIRE_LOG(Warning, ss.str());
      } else {
        for (auto &p : pools_) {
          p.strategy->deallocate(p.data);
          std::free(p.avail);
        }
      }
    }

    fixed_pool(const fixed_pool&) = delete;

    void* allocate(std::size_t bytes = 0) override final
    {
      // Check that bytes passed matches object_bytes_ or bytes was not passed (default = 0)
      UMPIRE_ASSERT(!bytes || bytes == object_bytes_);

      void *ptr = nullptr;

      for (auto it = pools_.rbegin(); it != pools_.rend(); ++it) {
        ptr = allocInPool(*it);
        if (ptr) {
          break;
        }
      }

      if (!ptr) {
        new_pool();
        ptr = allocate(bytes);
      }

      if (!ptr) {
        UMPIRE_ERROR("FixedPool::allocate(size=" << object_bytes_ << "): Could not allocate");
      }

      if constexpr(Tracking) {
        return this->track_allocation(this, ptr, object_bytes_);
      } else {
        return ptr;
      }
    }

    void deallocate(void* ptr) override final
    {
      for (auto& p : pools_) {
        const char* t_ptr = reinterpret_cast<char*>(ptr);
        const ptrdiff_t offset = t_ptr - p.data;
        if ((offset >= 0) && (offset < static_cast<ptrdiff_t>(bytes_data_))) {
          const std::size_t alloc_index = offset / object_bytes_;
          const std::size_t int_index   = alloc_index / bits_per_int;
          const short  bit_index   = alloc_index % bits_per_int;

          UMPIRE_ASSERT(! (p.avail[int_index] & (1 << bit_index)));

            // Flip bit 0 -> 1
          p.avail[int_index] ^= 1 << bit_index;
          p.num_avail++;

          if constexpr(Tracking) {
            this->untrack_allocation(ptr);
          }

          return;
        }
      }

      UMPIRE_ERROR("Could not find the pointer to deallocate");
  }

    std::size_t get_actual_size() const noexcept override final
    {
      return bytes_actual_;
    }

    camp::resources::Platform get_platform()
    {
      return memory_->get_platform();
    }

    bool pointerIsFromPool(void* ptr) const noexcept
    {
      for (const auto &p : pools_) {
        const char *t_ptr = reinterpret_cast<char *>(ptr);
        const ptrdiff_t offset = t_ptr - p.data;
        if ((offset >= 0) && (offset < static_cast<ptrdiff_t>(bytes_data_))) {
          return true;
        }
      }

      return false;
    }

    std::size_t numPools() const noexcept
    {
      return pools_.size();
    }

private:
    Memory* memory_;
    const std::size_t object_bytes_;
    const std::size_t objects_per_pool_;
    std::size_t bytes_actual_;
    std::size_t bytes_data_;
    std::size_t bytes_available_;
    std::vector<Pool> pools_;
};

} // end namespace strategy
} // end namespace umpire

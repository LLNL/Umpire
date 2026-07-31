//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-2025, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_strategy_binned_pool_HPP
#define UMPIRE_strategy_binned_pool_HPP

#include "umpire/strategy/allocation_strategy.hpp"
#include "umpire/util/error.hpp"

#include "fmt/format.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace umpire {
namespace strategy {

//! @brief Fast pool using power-of-2 size bins for O(1) allocation
//!
//! The binned_pool strategy provides O(1) allocation by using fixed power-of-2
//! size bins. Each bin maintains its own free list for allocations of that
//! size, and each bin can be configured with its own growth chunk size.
//!
//! @par Default Configuration
//! - Bins: 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 bytes
//! - Blocks per bin: 1024, 1024, 512, 256, 128, 64, 32, 16, 8
//! - Allocations larger than the largest bin are forwarded directly to parent
//!
//! @par Allocation Strategy
//! - O(1) bin lookup for the fixed number of bins
//! - Each bin has its own free list
//! - Empty bins are replenished by allocating another configured chunk
//! - A small aligned header stores the requested size and bin index
//!
//! @par Trade-offs
//! - Internal fragmentation: 65 bytes uses 128-byte bin (48% waste)
//! - Header overhead per allocation
//! - Not suitable for large allocations (> max bin size)
//! - Excellent for known small size ranges
//!
//! @tparam Memory The memory source type to wrap (must inherit from memory)
template<typename Memory>
class binned_pool : public allocation_strategy {
public:
  //! @brief Platform type propagated from wrapped memory source
  using platform = typename Memory::platform;
  //! @brief Number of fixed-size bins maintained by the pool.
  static constexpr std::size_t NUM_BINS = 9;
  //! @brief Array type used to configure per-bin sizes and chunk counts.
  using configuration_array = std::array<std::size_t, NUM_BINS>;

private:
  struct alignas(std::max_align_t) pool_header {
    std::size_t requested_size;
    std::size_t bin_index;
  };

  static_assert(alignof(pool_header) >= alignof(std::max_align_t),
                "binned_pool header must preserve standard alignment");

  static constexpr std::size_t HEADER_SIZE = sizeof(pool_header);
  static constexpr std::size_t DIRECT_ALLOC_INDEX = NUM_BINS;
  static constexpr std::size_t EAGER_BIN_COUNT = 5;

  configuration_array bin_sizes_;
  configuration_array blocks_per_bin_;
  std::array<std::vector<void*>, NUM_BINS> free_lists_;
  std::vector<void*> chunks_;

  std::size_t total_allocated_;
  std::size_t user_allocated_;
  std::array<std::size_t, NUM_BINS> bin_allocations_;

  static constexpr bool is_power_of_two(std::size_t value)
  {
    return value != 0 && (value & (value - 1)) == 0;
  }

  static std::size_t get_bin_index(std::size_t size, const configuration_array& bin_sizes)
  {
    for (std::size_t i = 0; i < NUM_BINS; ++i) {
      if (size <= bin_sizes[i]) {
        return i;
      }
    }

    return DIRECT_ALLOC_INDEX;
  }

  std::size_t get_bin_index(std::size_t size) const
  {
    return get_bin_index(size, bin_sizes_);
  }

  void validate_configuration() const
  {
    for (std::size_t i = 0; i < NUM_BINS; ++i) {
      if (!is_power_of_two(bin_sizes_[i])) {
        throw std::invalid_argument(
          fmt::format("binned_pool: bin size {} at index {} must be a non-zero power of two",
                      bin_sizes_[i], i));
      }

      if (i > 0 && bin_sizes_[i] <= bin_sizes_[i - 1]) {
        throw std::invalid_argument(
          fmt::format("binned_pool: bin size {} at index {} must be strictly greater than {}",
                      bin_sizes_[i], i, bin_sizes_[i - 1]));
      }

      if (blocks_per_bin_[i] == 0) {
        throw std::invalid_argument(
          fmt::format("binned_pool: blocks_per_bin for index {} must be greater than 0", i));
      }

      if (blocks_per_bin_[i] >
          std::numeric_limits<std::size_t>::max() / (bin_sizes_[i] + HEADER_SIZE)) {
        throw std::invalid_argument(
          fmt::format("binned_pool: chunk configuration for bin {} overflows size_t", i));
      }
    }
  }

  void allocate_chunk(std::size_t bin_index)
  {
    const std::size_t alloc_size = bin_sizes_[bin_index] + HEADER_SIZE;
    const std::size_t objects_per_chunk = blocks_per_bin_[bin_index];
    const std::size_t chunk_size = objects_per_chunk * alloc_size;
    void* chunk = parent_->allocate(chunk_size);

    if (!chunk) {
      UMPIRE_ERROR(out_of_memory_error,
                   fmt::format("binned_pool: failed to allocate chunk of {} bytes", chunk_size));
    }

    chunks_.push_back(chunk);
    total_allocated_ += chunk_size;

    auto& free_list = free_lists_[bin_index];
    free_list.reserve(free_list.size() + objects_per_chunk);

    char* base = static_cast<char*>(chunk);
    for (std::size_t i = 0; i < objects_per_chunk; ++i) {
      free_list.push_back(base + (i * alloc_size));
    }
  }

  void* store_header(void* ptr, std::size_t bin_index, std::size_t requested_size)
  {
    auto* header = static_cast<pool_header*>(ptr);
    header->requested_size = requested_size;
    header->bin_index = bin_index;
    return static_cast<void*>(header + 1);
  }

  static const pool_header* get_header(const void* user_ptr)
  {
    return reinterpret_cast<const pool_header*>(static_cast<const char*>(user_ptr) - HEADER_SIZE);
  }

  static void* get_alloc_ptr(void* user_ptr)
  {
    return static_cast<void*>(static_cast<char*>(user_ptr) - HEADER_SIZE);
  }

  static double calculate_fragmentation(
    std::size_t size,
    const configuration_array& bin_sizes)
  {
    const std::size_t bin_index = get_bin_index(size, bin_sizes);
    if (bin_index == DIRECT_ALLOC_INDEX) {
      return 0.0;
    }

    const std::size_t bin_size = bin_sizes[bin_index];
    return static_cast<double>(bin_size - size) / static_cast<double>(bin_size);
  }

public:
  //! @brief Return the default power-of-two bin sizes.
  static constexpr configuration_array default_bin_sizes()
  {
    return {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
  }

  //! @brief Return the default per-bin chunk lengths.
  static constexpr configuration_array default_blocks_per_bin()
  {
    return {1024, 1024, 512, 256, 128, 64, 32, 16, 8};
  }

  /*!
   * \brief Construct a quick pool with optional custom bin geometry.
   *
   * \param name Name for this pool instance.
   * \param parent Memory source to wrap.
   * \param bin_sizes Strictly increasing power-of-two bin sizes.
   * \param blocks_per_bin Number of blocks to allocate when growing each bin.
   *
   * \throws std::invalid_argument if the configuration is inconsistent.
   */
  explicit binned_pool(
    const std::string& name,
    Memory* parent,
    configuration_array bin_sizes = default_bin_sizes(),
    configuration_array blocks_per_bin = default_blocks_per_bin())
    : allocation_strategy(name, parent)
    , bin_sizes_(std::move(bin_sizes))
    , blocks_per_bin_(std::move(blocks_per_bin))
    , total_allocated_(0)
    , user_allocated_(0)
  {
    validate_configuration();
    bin_allocations_.fill(0);

    for (std::size_t i = 0; i < std::min(EAGER_BIN_COUNT, NUM_BINS); ++i) {
      free_lists_[i].reserve(blocks_per_bin_[i]);
      allocate_chunk(i);
    }
  }

  //! @brief Destructor that returns all backing chunks to the parent.
  ~binned_pool()
  {
    for (void* chunk : chunks_) {
      parent_->deallocate(chunk);
    }
  }

  /*!
   * \brief Allocate storage from the nearest fitting bin.
   *
   * Requests larger than the biggest configured bin are forwarded directly to
   * the parent after adding internal header storage.
   *
   * \param size Number of user-visible bytes to allocate.
   * \return Pointer to user storage, or `nullptr` for a zero-byte request.
   */
  void* allocate(std::size_t size) override
  {
    if (size == 0) {
      return nullptr;
    }

    if (size > std::numeric_limits<std::size_t>::max() - HEADER_SIZE) {
      UMPIRE_ERROR(out_of_memory_error,
                   fmt::format("binned_pool: allocation size {} overflows header accounting", size));
    }

    const std::size_t bin_index = get_bin_index(size);

    if (bin_index == DIRECT_ALLOC_INDEX) {
      const std::size_t alloc_size = size + HEADER_SIZE;
      void* ptr = parent_->allocate(alloc_size);
      if (!ptr) {
        UMPIRE_ERROR(out_of_memory_error,
                     fmt::format("binned_pool: failed to allocate {} bytes", alloc_size));
      }

      total_allocated_ += alloc_size;
      user_allocated_ += size;
      return store_header(ptr, DIRECT_ALLOC_INDEX, size);
    }

    auto& free_list = free_lists_[bin_index];
    if (free_list.empty()) {
      allocate_chunk(bin_index);
    }

    void* ptr = free_list.back();
    free_list.pop_back();

    ++bin_allocations_[bin_index];
    user_allocated_ += size;

    return store_header(ptr, bin_index, size);
  }

  /*!
   * \brief Deallocate storage previously returned by this pool.
   *
   * \param user_ptr Pointer to user storage. `nullptr` is a no-op.
   *
   * \throws std::runtime_error if the internal allocation header is invalid.
   */
  void deallocate(void* user_ptr) override
  {
    if (!user_ptr) {
      return;
    }

    const pool_header* header = get_header(user_ptr);
    const std::size_t requested_size = header->requested_size;
    const std::size_t bin_index = header->bin_index;
    void* alloc_ptr = get_alloc_ptr(user_ptr);

    user_allocated_ -= requested_size;

    if (bin_index == DIRECT_ALLOC_INDEX) {
      total_allocated_ -= requested_size + HEADER_SIZE;
      parent_->deallocate(alloc_ptr);
      return;
    }

    if (bin_index < NUM_BINS) {
      free_lists_[bin_index].push_back(alloc_ptr);
      --bin_allocations_[bin_index];
      return;
    }

    throw std::runtime_error(
      fmt::format("binned_pool: invalid bin index {} for pointer {:p}", bin_index, user_ptr));
  }

  //! @brief No-op release hook; binned_pool does not track fully free chunks.
  void release()
  {
    // No-op: tracking completely free chunks would add more metadata than this
    // simple fast-path pool intends to maintain.
  }

  //! @brief Return total bytes reserved from the parent, including headers.
  std::size_t get_total_allocated() const { return total_allocated_; }
  //! @brief Return live user-visible bytes currently allocated.
  std::size_t get_user_allocated() const { return user_allocated_; }
  //! @brief Return the number of backing chunks allocated across all bins.
  std::size_t get_chunk_count() const { return chunks_.size(); }

  //! @brief Return the default size of the bin at `index`, or 0 if out of range.
  static std::size_t get_bin_size(std::size_t index)
  {
    const auto bins = default_bin_sizes();
    return index < NUM_BINS ? bins[index] : 0;
  }

  //! @brief Return the configured size of the bin at `index`, or 0 if out of range.
  std::size_t get_configured_bin_size(std::size_t index) const
  {
    return index < NUM_BINS ? bin_sizes_[index] : 0;
  }

  //! @brief Return the configured chunk length of the bin at `index`, or 0 if out of range.
  std::size_t get_blocks_per_bin(std::size_t index) const
  {
    return index < NUM_BINS ? blocks_per_bin_[index] : 0;
  }

  //! @brief Return the compile-time number of bins.
  static constexpr std::size_t get_num_bins() { return NUM_BINS; }

  //! @brief Return the current live allocation count in the bin at `index`.
  std::size_t get_bin_allocations(std::size_t index) const
  {
    return index < NUM_BINS ? bin_allocations_[index] : 0;
  }

  //! @brief Return the current free-list length of the bin at `index`.
  std::size_t get_bin_free_count(std::size_t index) const
  {
    return index < NUM_BINS ? free_lists_[index].size() : 0;
  }

  //! @brief Compute internal fragmentation against the default bin layout.
  static double calculate_fragmentation(std::size_t size)
  {
    return calculate_fragmentation(size, default_bin_sizes());
  }

  //! @brief Compute internal fragmentation against this pool's configured bins.
  double calculate_configured_fragmentation(std::size_t size) const
  {
    return calculate_fragmentation(size, bin_sizes_);
  }
};

} // namespace strategy
} // namespace umpire

#endif // UMPIRE_strategy_binned_pool_HPP

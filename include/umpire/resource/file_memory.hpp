//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_resource_file_memory_HPP
#define UMPIRE_resource_file_memory_HPP

#include "umpire/config.hpp"

#if defined(UMPIRE_ENABLE_FILE_RESOURCE)

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "fmt/format.h"
#include "umpire/memory_resource.hpp"
#include "umpire/platform.hpp"
#include "umpire/resource/host_memory.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace resource {

/*!
 * \brief API v2 resource that backs allocations with `mmap`'d files.
 *
 * `file_memory` is a port of the legacy v1 `FileMemoryResource`. Each
 * allocation creates a new file (named `umpire_mem_<pid>_<counter>`) under a
 * directory selected by the `UMPIRE_MEMORY_FILE_DIR` environment variable
 * (defaulting to `./`), truncates it to a page-rounded size, and maps it with
 * `mmap`. Deallocation unmaps, closes, and removes the backing file.
 *
 * Unlike `host_memory`, per-allocation state (the backing file name and
 * mapped size) must be tracked alongside the pointer, so this resource does
 * not route allocation through the `Allocator` template parameter used by
 * `memory_resource`. It still inherits from `memory_resource<host_platform,
 * malloc_allocator, Tracking>` for the shared name/id/statistics plumbing and
 * platform reporting; the inherited `allocator_` member is unused since
 * `allocate()`/`deallocate()` are fully overridden to perform the mmap/munmap
 * sequence directly (mirroring how `null_resource` overrides both methods
 * around an allocator that is likewise never exercised for real storage).
 *
 * \tparam Tracking Whether allocations should be tracked in the shared v2
 *         registry.
 */
template<bool Tracking = true>
class file_memory : public memory_resource<host_platform, malloc_allocator, Tracking> {
private:
  using base = memory_resource<host_platform, malloc_allocator, Tracking>;

  //! \brief Per-process counter used to generate unique backing file names.
  static std::atomic<unsigned long>& file_counter() {
    static std::atomic<unsigned long> counter{0};
    return counter;
  }

  //! \brief Backing file name and mapped (page-rounded) size for each live allocation.
  std::unordered_map<void*, std::pair<std::string, std::size_t>> m_size_map;

  // Singleton instance (for default config only)
  static file_memory& instance() {
    static file_memory inst;
    return inst;
  }

  // Private constructor for singleton
  file_memory() : base("FILE") {}

public:
  //! \brief Return the process-wide default FILE resource singleton.
  static file_memory& get() {
    return instance();
  }

  /*!
   * \brief Construct a named file-backed resource.
   *
   * \param name Human-readable resource name.
   */
  explicit file_memory(const std::string& name) : base(name) {}

  /*!
   * \brief Release the backing files of any allocations still outstanding.
   */
  ~file_memory() {
    while (!m_size_map.empty()) {
      deallocate(m_size_map.begin()->first);
    }
  }

  file_memory(const file_memory&) = delete;
  file_memory& operator=(const file_memory&) = delete;

  /*!
   * \brief Create a file-backed mapping of size bytes using mmap.
   *
   * Mirrors the v1 `FileMemoryResource::allocate()` sequence:
   * 1) Resolve the output directory from `UMPIRE_MEMORY_FILE_DIR` (default `./`).
   * 2) Build a unique file name (`umpire_mem_<pid>_<counter>`) and `open()` it.
   * 3) Round the requested size up to a multiple of the system page size.
   * 4) `ftruncate()` the file to the rounded size.
   * 5) `mmap()` the file and record its name/size in `m_size_map`.
   *
   * \param bytes Requested number of bytes. Zero returns `nullptr`.
   * \return Pointer to the mapped storage, or `nullptr` for a zero-byte request.
   *
   * \throws runtime_error if opening, truncating, or mapping the backing file fails.
   */
  void* allocate(std::size_t bytes) override {
    if (bytes == 0) {
      return nullptr;  // Match malloc-family behavior for zero-size requests
    }

    // Find output file directory for mmap files
    const char* memory_file_dir{std::getenv("UMPIRE_MEMORY_FILE_DIR")};
    std::string default_dir = "./";
    if (memory_file_dir) {
      default_dir = memory_file_dir;
    }

    // Create name and open file
    std::stringstream ss;
    ss << default_dir << "umpire_mem_" << getpid() << "_" << file_counter().fetch_add(1, std::memory_order_relaxed);

    int fd{open(ss.str().c_str(), O_RDWR | O_CREAT, S_IRWXU)};
    if (fd == -1) {
      UMPIRE_ERROR(runtime_error, fmt::format("Opening file {} failed: {}", ss.str(), strerror(errno)));
    }

    // Round requested size up to a page boundary
    const std::size_t pagesize{static_cast<std::size_t>(sysconf(_SC_PAGE_SIZE))};
    std::size_t rounded_bytes{((bytes + (pagesize - 1)) / pagesize) * pagesize};

    // Truncate file
    int trunc_result{ftruncate(fd, static_cast<off_t>(rounded_bytes))};
    if (trunc_result == -1) {
      int errno_save = errno;
      ::close(fd);
      remove(ss.str().c_str());
      UMPIRE_ERROR(runtime_error, fmt::format("truncate of file {} failed: {}", ss.str(), strerror(errno_save)));
    }

    void* ptr{mmap(NULL, rounded_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)};
    if (ptr == MAP_FAILED) {
      int errno_save = errno;
      ::close(fd);
      remove(ss.str().c_str());
      UMPIRE_ERROR(runtime_error,
                   fmt::format("mmap of {} to file {} failed: {}", rounded_bytes, ss.str(), strerror(errno_save)));
    }

    // File descriptor is no longer needed once the mapping is established.
    ::close(fd);

    m_size_map.emplace(ptr, std::make_pair(ss.str(), rounded_bytes));

    if constexpr (Tracking) {
      base::track_allocation(ptr, bytes);
    }

    return ptr;
  }

  /*!
   * \brief Unmap and remove the file backing a previous allocation.
   *
   * \param ptr Pointer previously returned by `allocate()`. `nullptr` is a
   *        no-op. Pointers not known to this resource are also ignored.
   *
   * \throws runtime_error if `munmap()` or `remove()` of the backing file fails.
   */
  void deallocate(void* ptr) override {
    if (!ptr) return;  // nullptr deallocation is safe no-op

    auto iter = m_size_map.find(ptr);
    if (iter == m_size_map.end()) {
      return;  // Unknown pointer: safe no-op, mirrors other v2 resources
    }

    const std::string file_name = iter->second.first;
    const std::size_t mapped_size = iter->second.second;

    if (munmap(ptr, mapped_size) < 0) {
      UMPIRE_ERROR(runtime_error, fmt::format("munmap of file {} failed: {}", file_name, strerror(errno)));
    }

    if (remove(file_name.c_str()) < 0) {
      UMPIRE_ERROR(runtime_error, fmt::format("remove of file {} failed: {}", file_name, strerror(errno)));
    }

    if constexpr (Tracking) {
      base::untrack_allocation(ptr);
    }

    m_size_map.erase(iter);
  }
};

//! \brief Tracking-enabled file resource.
using default_file_memory = file_memory<true>;
//! \brief File resource alias with tracking disabled for low-overhead paths.
using fast_file_memory = file_memory<false>;

} // namespace resource
} // namespace umpire

#endif // UMPIRE_ENABLE_FILE_RESOURCE

#endif // UMPIRE_resource_file_memory_HPP

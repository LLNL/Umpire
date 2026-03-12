//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef REPLAY_ReplayFile_HPP
#define REPLAY_ReplayFile_HPP
#if !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)

#include <string>
#include <vector>
#include "ReplayOptions.hpp"
#include "umpire/Allocator.hpp"

class ReplayFile {
public:
  enum rtype {
      INIT = 0
    , MEMORY_RESOURCE = 1
    , ALLOCATION_ADVISOR = 2
    , DYNAMIC_POOL_LIST = 3
    , MONOTONIC = 4
    , NAMED = 5
    , SLOT_POOL = 6
    , SIZE_LIMITER = 7
    , THREADSAFE_ALLOCATOR = 8
    , FIXED_POOL = 9
    , MIXED_POOL = 10
    , ALLOCATION_PREFETCHER = 11
    , NUMA_POLICY = 12
    , QUICKPOOL = 13
  };

  static const std::size_t max_allocators{256 * 1024};
  static const std::size_t max_name_length{512};

  struct AllocatorTableEntry {
    rtype type{rtype::INIT};
    std::size_t line_number{0};    // Causal line number of input file
    bool introspection{0};
    char name[max_name_length]{0};
    char base_name[max_name_length]{0};
    int argc{0};
    union {
      struct {
        int device_id{0};
        char advice[max_name_length]{0};
        char accessing_allocator[max_name_length]{0};
      } advisor;
      struct {
        int node{0};
      } numa;
      struct {
        std::size_t initial_alloc_size{0};
        std::size_t min_alloc_size{0};
        int alignment{0};
      } pool;
      struct {
        std::size_t capacity{0};
      } monotonic_pool;
      struct {
        std::size_t slots{0};
      } slot_pool;
      struct {
        std::size_t size_limit{0};
      } size_limiter;
      struct {
        std::size_t object_bytes{0};
        std::size_t objects_per_pool{0};
      } fixed_pool;
      struct {
        std::size_t smallest_fixed_blocksize{0};
        std::size_t largest_fixed_blocksize{0};
        std::size_t max_fixed_blocksize{0};
        std::size_t size_multiplier{0};
        std::size_t dynamic_initial_alloc_bytes{0};
        std::size_t dynamic_min_alloc_bytes{0};
        std::size_t dynamic_align_bytes{0};
      } mixed_pool;
    } argv;
    umpire::Allocator* allocator{nullptr};
  };

  enum otype {
      ALLOCATOR_CREATION = 1
    , ALLOCATE
    , COALESCE
    , COPY
    , MOVE
    , DEALLOCATE
    , REALLOCATE
    , REALLOCATE_EX
    , RELEASE
    , SETDEFAULTALLOCATOR
  };

  struct Operation {
    otype       op_type;
    std::size_t op_line_number;     // Causal line number of input file
    int         op_allocator;
    void*       op_allocated_ptr;
    std::size_t op_size;            // Size of allocation/operation
    std::size_t op_offsets[2];      // 0-src, 1-dst
    std::size_t op_alloc_ops[2];    // 0-src, 1-dst/prev
  };

  const uint64_t REPLAY_MAGIC =
    static_cast<uint64_t>(
            static_cast<uint64_t>(0x7f) << 48
          | static_cast<uint64_t>('R') << 40
          | static_cast<uint64_t>('E') << 32
          | static_cast<uint64_t>('P') << 24
          | static_cast<uint64_t>('L') << 16
          | static_cast<uint64_t>('A') << 8
          | static_cast<uint64_t>('Y'));

  const uint64_t REPLAY_VERSION = 17;

  struct Header {
    struct Magic {
      uint64_t magic;
      uint64_t version;
    } m;
    std::size_t num_allocators;
    std::size_t num_operations;
    AllocatorTableEntry allocators[max_allocators];
    Operation ops[1];
  };

  ReplayFile( const ReplayOptions& options );
  ~ReplayFile( );
  ReplayFile::Header* getOperationsTable();

  void copyString(std::string source, char (&dest)[max_name_length]);
  bool compileNeeded() { return m_compile_needed; }
  std::string getLine(std::size_t lineno);
  std::string getInputFileName() { return m_options.input_file; }

private:
  ReplayOptions m_options;
  Header* m_op_tables{nullptr};
  const std::string m_binary_filename;
  int m_fd;
  bool m_compile_needed{false};
  off_t max_file_size{0};

  void checkHeader();
};
#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
#endif // REPLAY_ReplayFile_HPP

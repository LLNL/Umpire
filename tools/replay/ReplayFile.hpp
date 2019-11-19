//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef REPLAY_ReplayFile_HPP
#define REPLAY_ReplayFile_HPP

#include <string>
#include <vector>
#include "umpire/Allocator.hpp"

class ReplayFile {
public:
  enum rtype {
      MEMORY_RESOURCE = 1
    , ALLOCATION_ADVISOR
    , DYNAMIC_POOL_LIST
    , DYNAMIC_POOL_MAP
    , MONOTONIC
    , SLOT_POOL
    , SIZE_LIMITER
    , THREADSAFE_ALLOCATOR
    , FIXED_POOL
    , MIXED_POOL
    , ALLOCATION_PREFETCHER
  };

  static const std::size_t max_allocators{512};
  static const std::size_t max_name_length{256};

  struct AllocatorTableEntry {
    rtype type;
    bool introspection;
    char name[max_name_length];
    char base_name[max_name_length];
    int argc;
    union {
      struct {
        int device_id;
        char advice[max_name_length];
        char accessing_allocator[max_name_length];
      } advisor;
      struct {
        std::size_t initial_alloc_size;
        std::size_t min_alloc_size;
      } dynamic_pool_list;
      struct {
        std::size_t initial_alloc_size;
        std::size_t min_alloc_size;
        int alignment;
      } dynamic_pool_map;
      struct {
        std::size_t capacity;
      } monotonic_pool;
      struct {
        std::size_t slots;
      } slot_pool;
      struct {
        std::size_t size_limit;
      } size_limiter;
      struct {
        std::size_t object_bytes;
        std::size_t objects_per_pool;
      } fixed_pool;
      struct {
        std::size_t smallest_fixed_blocksize;
        std::size_t largest_fixed_blocksize;
        std::size_t max_fixed_blocksize;
        std::size_t size_multiplier;
        std::size_t dynamic_initial_alloc_bytes;
        std::size_t dynamic_min_alloc_bytes;
        std::size_t dynamic_align_bytes;
      } mixed_pool;
    } argv;
    umpire::Allocator* allocator{nullptr};
  };

  enum otype {
      ALLOCATOR_CREATION = 1
    , ALLOCATE
    , DEALLOCATE
    , COALESCE
    , RELEASE
  };

  struct Operation {
    otype type;
    int allocator_table_index;

    union {
      struct {
        std::size_t size;
        void* ptr;
      } allocate;
      struct {
        int allocation_op_idx;   // Index to actual allocation operation to get ptr
      } deallocate ;
    } argv ;
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

  const uint64_t REPLAY_VERSION = 3;

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

  ReplayFile( std::string input_filename, std::string binary_filename );
  ~ReplayFile( );
  ReplayFile::Header* getOperationsTable();

  void copyString(std::string source, char (&dest)[max_name_length]);
  bool compileNeeded() { return m_compile_needed; }

private:
  Header* m_op_tables{nullptr};
  const std::string m_input_filename;
  const std::string m_binary_filename;
  int m_fd;
  bool m_compile_needed{false};
  off_t max_file_size{0};

  void checkHeader();
};

#endif // REPLAY_ReplayFile_HPP

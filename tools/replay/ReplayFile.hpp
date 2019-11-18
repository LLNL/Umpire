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
      } advisor ;
      struct {
        std::size_t initial_alloc_size;
        std::size_t min_alloc_size;
      } dynamic_pool_list ;
      struct {
        std::size_t initial_alloc_size;
        std::size_t min_alloc_size;
        int alignment;
      } dynamic_pool_map ;
      struct {
        std::size_t capacity;
      } monotonic_pool ;
      struct {
        std::size_t slots;
      } slot_pool ;
      struct {
        std::size_t size_limit;
      } size_limiter ;
      struct {
        std::size_t object_bytes;
        std::size_t objects_per_pool;
      } fixed_pool ;
      struct {
        std::size_t smallest_fixed_blocksize;
        std::size_t largest_fixed_blocksize;
        std::size_t max_fixed_blocksize;
        std::size_t size_multiplier;
        std::size_t dynamic_initial_alloc_bytes;
        std::size_t dynamic_min_alloc_bytes;
        std::size_t dynamic_align_bytes;
      } mixed_pool ;
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

  const int header_version = 1;
  struct Header {
    int version{2};
    std::size_t num_allocators{0};
    std::size_t num_operations{0};
    AllocatorTableEntry allocators[max_allocators];
    Operation ops[1];
  };

  ReplayFile( std::string in_file_name );
  ~ReplayFile( );
  ReplayFile::Header* getOperationsTable();

  void copyString(std::string source, char (&dest)[max_name_length]);
  bool compileNeeded() { return m_compile_needed; }

private:
  const std::string m_bin_suffix{".replaybin"};
  Header* m_op_tables{nullptr};
  const std::string m_in_file_name;
  const std::string m_bin_file_name;
  int m_fd;
  bool m_compile_needed{false};
  off_t max_file_size{0};
};

#endif // REPLAY_ReplayFile_HPP

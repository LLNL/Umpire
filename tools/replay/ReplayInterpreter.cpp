//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <fstream>
#include <sstream>
#include <string>

#include "ReplayInterpreter.hpp"
#include "ReplayMacros.hpp"
#include "ReplayOperationManager.hpp"
#include "ReplayFile.hpp"
#include "umpire/tpl/json/json.hpp"

#if !defined(_MSC_VER)
#include <cxxabi.h>
#endif

void ReplayInterpreter::runOperations(bool gather_statistics)
{
  ReplayOperationManager m_operation_mgr{m_ops.getOperationsTable()};

  m_operation_mgr.runOperations(gather_statistics);
}

void ReplayInterpreter::buildOperations()
{
  while ( std::getline(m_input_file, m_line) ) {
    const std::string header("{ \"kind\":\"replay\", \"uid\":");
    auto const header_len(header.size());

    if ( m_line.size() <= header_len || m_line.substr(0, header_len) != header.substr(0, header_len) )
      continue;

    m_json.clear();
    m_json = nlohmann::json::parse(m_line);

    if (   m_json["event"] == "allocation_map_insert" 
        || m_json["event"] == "allocation_map_find"
        || m_json["event"] == "allocation_map_remove"
        || m_json["event"] == "allocation_map_clear"
    ) {
      continue;
    }

    if ( m_json["event"] == "makeAllocator" ) {
      replay_compileAllocator();
    }
    else if ( m_json["event"] == "makeMemoryResource" ) {
      replay_compileMemoryResource();
    }
    else if ( m_json["event"] == "allocate" ) {
      replay_compileAllocate();
    }
    else if ( m_json["event"] == "deallocate" ) {
      replay_compileDeallocate();
    }
    else if ( m_json["event"] == "coalesce" ) {
      replay_compileCoalesce();
    }
    else if ( m_json["event"] == "release" ) {
      replay_compileRelease();
    }
    else if ( m_json["event"] == "version" ) {
      m_log_version_major = m_json["payload"]["major"];
      m_log_version_minor = m_json["payload"]["minor"];
      m_log_version_patch = m_json["payload"]["patch"];

      if (   m_log_version_major != UMPIRE_VERSION_MAJOR
          || m_log_version_minor != UMPIRE_VERSION_MINOR
          || m_log_version_patch != UMPIRE_VERSION_PATCH ) {

        REPLAY_WARNING("Warning, version mismatch:\n"
          << "  Tool version: " << UMPIRE_VERSION_MAJOR << "." << UMPIRE_VERSION_MINOR << "." << UMPIRE_VERSION_PATCH << std::endl
          << "  Log  version: "
          << m_log_version_major << "."
          << m_log_version_minor  << "."
          << m_log_version_patch);

        if (m_json["payload"]["major"] != UMPIRE_VERSION_MAJOR) {
          REPLAY_WARNING("Warning, major version mismatch - attempting replay anyway...\n"
            << "  Tool version: " << UMPIRE_VERSION_MAJOR << "." << UMPIRE_VERSION_MINOR << "." << UMPIRE_VERSION_PATCH << std::endl
            << "  Log  version: "
            << m_log_version_major << "."
            << m_log_version_minor  << "."
            << m_log_version_patch);
        }
      }
    }
    else {
      REPLAY_ERROR("Unknown Replay (" << m_json["event"] << ")");
    }
  }
}

//
// Return: > 0 success, 0 eof, < 0 error
//
int ReplayInterpreter::getSymbolicOperation( std::string& , std::string& )
{
  return 0;   // EOF
}

ReplayInterpreter::ReplayInterpreter( std::string in_file_name ):
    m_input_file_name{in_file_name}, m_input_file{in_file_name}, m_ops{in_file_name}
{
  if ( ! m_input_file.is_open() )
    REPLAY_ERROR("Unable to open input file " << in_file_name);
}

template <typename T> void get_from_string( const std::string& s, T& val )
{
    std::istringstream ss(s);
    ss >> val;
}

void ReplayInterpreter::strip_off_base(std::string& s)
{
  const std::string base("_base");

  if (s.length() > base.length()) {
    if (s.compare(s.length() - base.length(), base.length(), base) == 0) {
      s.erase(s.length() - base.length(), base.length());
    }
  }
}

void ReplayInterpreter::replay_compileMemoryResource( void )
{
  const std::string allocator_name{m_json["payload"]["name"]};
  const std::string obj_s{m_json["result"]};
  const uint64_t obj_p{std::stoul(obj_s, nullptr, 0)};
  ReplayFile::Header* hdr = m_ops.getOperationsTable();

  m_allocator_indices[obj_p] = hdr->num_allocators;

  ReplayFile::AllocatorTableEntry* alloc = &(hdr->allocators[hdr->num_allocators]);

  alloc->type = ReplayFile::rtype::MEMORY_RESOURCE;
  alloc->introspection = false;
  alloc->argc = 0;
  m_ops.copyString(allocator_name, alloc->name);

  ReplayFile::Operation* op = &hdr->ops[hdr->num_operations];
  op->type = ReplayFile::otype::ALLOCATOR_CREATION;
  op->allocator_idx = hdr->num_allocators;
  m_allocator_index[allocator_name] = hdr->num_allocators;

  hdr->num_allocators++;
  hdr->num_operations++;
}

void ReplayInterpreter::replay_compileAllocator( void )
{
  ReplayFile::Header* hdr = m_ops.getOperationsTable();

  ReplayFile::AllocatorTableEntry* alloc = 
            & (m_ops.getOperationsTable()->allocators[hdr->num_allocators]);

  const std::string allocator_name{m_json["payload"]["allocator_name"]};

  m_ops.copyString(allocator_name, alloc->name);

  if ( m_json["result"].is_null() ) {
    const bool introspection{m_json["payload"]["with_introspection"]};
    const std::string raw_mangled_type{m_json["payload"]["type"]};

    alloc->introspection = introspection;
    alloc->argc = static_cast<int>(m_json["payload"]["args"].size());

    std::string type;
    if (m_log_version_major >= 2) {
      const std::string type_prefix{raw_mangled_type.substr(0, 2)};

      // Add _Z so that we can demangle the external symbol
      const std::string mangled_type = 
        (type_prefix == "_Z") ? raw_mangled_type : std::string{"_Z"} + raw_mangled_type;

      auto result = abi::__cxa_demangle(
          mangled_type.c_str(),
          nullptr,
          nullptr,
          nullptr);
      if (!result) {
          REPLAY_ERROR("Failed to demangle strategy type. Mangled type: " << mangled_type);
      }
      type = std::string{result};
      ::free(result);
    } else {
      type = raw_mangled_type;
    }

    if ( type == "umpire::strategy::AllocationAdvisor" ) {
      const std::string base_allocator_name{m_json["payload"]["args"][0]};
      const std::string advice_operation {m_json["payload"]["args"][1]};
      const std::string last_arg{m_json["payload"]["args"][alloc->argc - 1]};

      int device_id{-1};   // Use default argument if negative
      if (last_arg.find_first_not_of( "0123456789" ) == std::string::npos) {
        std::stringstream ss(last_arg);
        ss >> device_id;
      }

      alloc->type = ReplayFile::rtype::ALLOCATION_ADVISOR;
      m_ops.copyString(base_allocator_name, alloc->base_name);
      m_ops.copyString(advice_operation, alloc->argv.advisor.advice);
      alloc->argv.advisor.device_id = device_id;

      if (device_id >= 0) { // Optional device ID specified
        switch ( alloc->argc ) {
        default:
          REPLAY_ERROR("Invalid number of arguments (" << alloc->argc
            << " for " << type << " operation.  Stopping");
        case 3:
          break;
        case 4:
          const std::string accessing_allocator_name{m_json["payload"]["args"][2]};
          m_ops.copyString(accessing_allocator_name, alloc->argv.advisor.accessing_allocator);
          break;
        }
      }
      else { // Use default device_id
        switch ( alloc->argc ) {
        default:
          REPLAY_ERROR("Invalid number of arguments (" << alloc->argc
            << " for " << type << " operation.  Stopping");
        case 2:
          break;
        case 3:
          const std::string accessing_allocator_name{m_json["payload"]["args"][2]};
          m_ops.copyString(accessing_allocator_name, alloc->argv.advisor.accessing_allocator);
          break;
        }
      }
    }
    else if ( type == "umpire::strategy::DynamicPoolList" ) {
      const std::string base_allocator_name{m_json["payload"]["args"][0]};

      alloc->type = ReplayFile::rtype::DYNAMIC_POOL_LIST;
      m_ops.copyString(base_allocator_name, alloc->base_name);

      // Now grab the optional fields
      if (alloc->argc >= 3) {
        get_from_string(m_json["payload"]["args"][1],
                        alloc->argv.dynamic_pool_list.initial_alloc_size);
        get_from_string(m_json["payload"]["args"][2],
                        alloc->argv.dynamic_pool_list.min_alloc_size);
      }
      else if (alloc->argc == 2) {
        get_from_string(m_json["payload"]["args"][1],
                        alloc->argv.dynamic_pool_list.initial_alloc_size);
      }
    }
    else if ( type == "umpire::strategy::DynamicPool" || type == "umpire::strategy::DynamicPoolMap" ) {
      const std::string base_allocator_name{m_json["payload"]["args"][0]};

      alloc->type = ReplayFile::rtype::DYNAMIC_POOL_MAP;
      m_ops.copyString(base_allocator_name, alloc->base_name);

      if (alloc->argc >= 4) {
        get_from_string(m_json["payload"]["args"][1],
                        alloc->argv.dynamic_pool_map.initial_alloc_size);
        get_from_string(m_json["payload"]["args"][2],
                        alloc->argv.dynamic_pool_map.min_alloc_size);
        get_from_string(m_json["payload"]["args"][3],
                        alloc->argv.dynamic_pool_map.alignment);
      }
      else if (alloc->argc >= 3) {
        get_from_string(m_json["payload"]["args"][1], alloc->argv.dynamic_pool_map.initial_alloc_size);
        get_from_string(m_json["payload"]["args"][2], alloc->argv.dynamic_pool_map.min_alloc_size);
      }
      else if (alloc->argc == 2) {
        get_from_string(m_json["payload"]["args"][1],
            alloc->argv.dynamic_pool_map.initial_alloc_size);
      }
    }
    else if ( type == "umpire::strategy::MonotonicAllocationStrategy" ) {
      const std::string base_allocator_name{m_json["payload"]["args"][0]};

      alloc->type = ReplayFile::rtype::MONOTONIC;
      m_ops.copyString(base_allocator_name, alloc->base_name);

      get_from_string(m_json["payload"]["args"][1], 
                      alloc->argv.monotonic_pool.capacity);
    }
    else if ( type == "umpire::strategy::SlotPool" ) {
      const std::string base_allocator_name{m_json["payload"]["args"][0]};

      alloc->type = ReplayFile::rtype::SLOT_POOL;
      m_ops.copyString(base_allocator_name, alloc->base_name);
      get_from_string(m_json["payload"]["args"][1], alloc->argv.slot_pool.slots);
    }
    else if ( type == "umpire::strategy::SizeLimiter" ) {
      const std::string base_allocator_name{m_json["payload"]["args"][0]};

      alloc->type = ReplayFile::rtype::SIZE_LIMITER;
      m_ops.copyString(base_allocator_name, alloc->base_name);
      get_from_string(m_json["payload"]["args"][1], alloc->argv.size_limiter.size_limit);
    }
    else if ( type == "umpire::strategy::ThreadSafeAllocator" ) {
      const std::string base_allocator_name{m_json["payload"]["args"][0]};

      alloc->type = ReplayFile::rtype::THREADSAFE_ALLOCATOR;
      m_ops.copyString(base_allocator_name, alloc->base_name);
    }
    else if ( type == "umpire::strategy::FixedPool" ) {
      const std::string base_allocator_name{m_json["payload"]["args"][0]};

      alloc->type = ReplayFile::rtype::FIXED_POOL;
      m_ops.copyString(base_allocator_name, alloc->base_name);
      get_from_string(m_json["payload"]["args"][1], alloc->argv.fixed_pool.object_bytes);

      // Now grab the optional fields
      if (alloc->argc >= 3) {
        get_from_string(m_json["payload"]["args"][2], alloc->argv.fixed_pool.objects_per_pool);
      }
    }
    else if ( type == "umpire::strategy::MixedPool" ) {
      const std::string base_allocator_name{m_json["payload"]["args"][0]};

      alloc->type = ReplayFile::rtype::MIXED_POOL;
      m_ops.copyString(base_allocator_name, alloc->base_name);

      // Now grab the optional fields
      if (alloc->argc >= 8) {
        get_from_string(m_json["payload"]["args"][1], alloc->argv.mixed_pool.smallest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][2], alloc->argv.mixed_pool.largest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][3], alloc->argv.mixed_pool.max_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][4], alloc->argv.mixed_pool.size_multiplier);
        get_from_string(m_json["payload"]["args"][5], alloc->argv.mixed_pool.dynamic_initial_alloc_bytes);
        get_from_string(m_json["payload"]["args"][6], alloc->argv.mixed_pool.dynamic_min_alloc_bytes);
        get_from_string(m_json["payload"]["args"][7], alloc->argv.mixed_pool.dynamic_align_bytes);
      }
      else if (alloc->argc >= 7) {
        get_from_string(m_json["payload"]["args"][1], alloc->argv.mixed_pool.smallest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][2], alloc->argv.mixed_pool.largest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][3], alloc->argv.mixed_pool.max_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][4], alloc->argv.mixed_pool.size_multiplier);
        get_from_string(m_json["payload"]["args"][5], alloc->argv.mixed_pool.dynamic_initial_alloc_bytes);
        get_from_string(m_json["payload"]["args"][6], alloc->argv.mixed_pool.dynamic_min_alloc_bytes);
      }
      else if (alloc->argc >= 6) {
        get_from_string(m_json["payload"]["args"][1], alloc->argv.mixed_pool.smallest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][2], alloc->argv.mixed_pool.largest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][3], alloc->argv.mixed_pool.max_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][4], alloc->argv.mixed_pool.size_multiplier);
        get_from_string(m_json["payload"]["args"][5], alloc->argv.mixed_pool.dynamic_initial_alloc_bytes);
      }
      else if (alloc->argc >= 5) {
        get_from_string(m_json["payload"]["args"][1], alloc->argv.mixed_pool.smallest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][2], alloc->argv.mixed_pool.largest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][3], alloc->argv.mixed_pool.max_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][4], alloc->argv.mixed_pool.size_multiplier);
      }
      else if (alloc->argc >= 4) {
        get_from_string(m_json["payload"]["args"][1], alloc->argv.mixed_pool.smallest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][2], alloc->argv.mixed_pool.largest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][3], alloc->argv.mixed_pool.max_fixed_blocksize);
      }
      else if (alloc->argc >= 3) {
        get_from_string(m_json["payload"]["args"][1], alloc->argv.mixed_pool.smallest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][2], alloc->argv.mixed_pool.largest_fixed_blocksize);
      }
      else if (alloc->argc >= 2) {
        get_from_string(m_json["payload"]["args"][1], alloc->argv.mixed_pool.smallest_fixed_blocksize);
      }
    }
    else {
      REPLAY_ERROR("Unknown class (" << type << "), skipping.");
    }
  }
  else {
    const std::string obj_s{m_json["result"]["allocator_ref"]};
    const uint64_t obj_p{std::stoul(obj_s, nullptr, 0)};

    m_allocator_indices[obj_p] = hdr->num_allocators;

    ReplayFile::Operation* op = &hdr->ops[hdr->num_operations];
    op->type = ReplayFile::otype::ALLOCATOR_CREATION;
    op->allocator_idx = hdr->num_allocators;

    m_allocator_index[allocator_name] = hdr->num_allocators;

    hdr->num_allocators++;
    hdr->num_operations++;
  }
}

void ReplayInterpreter::replay_compileAllocate( void )
{
  const std::string alloc_obj_s{m_json["payload"]["allocator_ref"]};
  const uint64_t alloc_obj_p{std::stoul(alloc_obj_s, nullptr, 0)};
  auto n_iter(m_allocator_indices.find(alloc_obj_p));

  if ( n_iter == m_allocator_indices.end() )
    REPLAY_ERROR("Unknown allocator " << (void*)alloc_obj_p);

  const AllocatorIndex& allocator_number{n_iter->second};

  ReplayFile::Header* hdr = m_ops.getOperationsTable();
  ReplayFile::Operation* op = &hdr->ops[hdr->num_operations];

  //
  // For allocations, two records are written.  The first record simply
  // contains the intention of the allocation (e.g. allocator and size).
  // The second record contains the result of the operation.
  //
  // We separate the intention from the result so that we may attempt to
  // replay an operation that may have failed during the initial run being
  // replayed.
  //
  if ( m_json["result"].is_null() ) {
    const std::size_t alloc_size{m_json["payload"]["size"]};

    op->type = ReplayFile::otype::ALLOCATE;
    op->allocator_idx = allocator_number;
    op->argv.allocate.size = alloc_size;
  }
  else {
    const std::string memory_str{m_json["result"]["memory_ptr"]};
    const uint64_t memory_ptr{std::stoul(memory_str, nullptr, 0)};

    m_allocation_id[memory_ptr] = hdr->num_operations;
    op->argv.allocate.ptr = nullptr;
    hdr->num_operations++;
  }
}

void ReplayInterpreter::replay_compileDeallocate( void )
{
  const std::string alloc_obj_s{m_json["payload"]["allocator_ref"]};
  const uint64_t alloc_obj_p{std::stoul(alloc_obj_s, nullptr, 0)};
  auto n_iter(m_allocator_indices.find(alloc_obj_p));

  if ( n_iter == m_allocator_indices.end() )
    REPLAY_ERROR("Unable to find allocator for: " << m_json["payload"]["memory_ptr"] << " deallocation ignored");

  const AllocatorIndex& allocator_number{n_iter->second};

  ReplayFile::Header* hdr = m_ops.getOperationsTable();
  ReplayFile::Operation* op = &hdr->ops[hdr->num_operations];

  op->type = ReplayFile::otype::DEALLOCATE;
  op->allocator_idx = allocator_number;

  const std::string memory_str{m_json["payload"]["memory_ptr"]};
  const uint64_t memory_ptr{std::stoul(memory_str, nullptr, 0)};

  op->argv.deallocate.allocation_idx = m_allocation_id[memory_ptr];
}

void ReplayInterpreter::replay_compileCoalesce( void )
{
  std::string allocator_name{m_json["payload"]["allocator_name"]};
  strip_off_base(allocator_name);

  ReplayFile::Header* hdr = m_ops.getOperationsTable();
  ReplayFile::Operation* op = &hdr->ops[hdr->num_operations];
  op->type = ReplayFile::otype::COALESCE;
  op->allocator_idx = m_allocator_index[allocator_name];
  hdr->num_operations++;
}

void ReplayInterpreter::replay_compileRelease( void )
{
  const std::string alloc_obj_s{m_json["payload"]["allocator_ref"]};
  const uint64_t alloc_obj_p{std::stoul(alloc_obj_s, nullptr, 0)};
  auto n_iter(m_allocator_indices.find(alloc_obj_p));
  if ( n_iter == m_allocator_indices.end() )
    REPLAY_ERROR("Unable to find allocator for: " << m_json["payload"]["memory_ptr"] << " release ignored");
  const AllocatorIndex& allocator_number{n_iter->second};

  ReplayFile::Header* hdr = m_ops.getOperationsTable();
  ReplayFile::Operation* op = &hdr->ops[hdr->num_operations];
  op->type = ReplayFile::otype::RELEASE;
  op->allocator_idx = allocator_number;
  hdr->num_operations++;
}


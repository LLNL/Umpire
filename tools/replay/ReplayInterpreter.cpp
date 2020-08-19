//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>

#if !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
#include <cxxabi.h>   // for __cxa_demangle

#include "ReplayInterpreter.hpp"
#include "ReplayMacros.hpp"
#include "ReplayOperationManager.hpp"
#include "ReplayFile.hpp"
#include "umpire/tpl/json/json.hpp"

ReplayInterpreter::ReplayInterpreter( std::string in_file_name ):
    m_input_file_name{in_file_name}, m_input_file{in_file_name}
{
  if ( ! m_input_file.is_open() )
    REPLAY_ERROR("Unable to open input file " << in_file_name);

  m_ops = new ReplayFile{m_input_file_name, m_input_file_name + ".bin" };
}

ReplayInterpreter::~ReplayInterpreter()
{
  if (m_ops != nullptr) {
    delete m_ops;
    m_ops = nullptr;
  }
}

void ReplayInterpreter::printInfo()
{
  ReplayOperationManager m_operation_mgr{m_ops->getOperationsTable()};

  m_operation_mgr.printInfo();
}

void ReplayInterpreter::runOperations(bool gather_statistics, bool skip_operations)
{
  ReplayOperationManager m_operation_mgr{m_ops->getOperationsTable()};

  m_operation_mgr.runOperations(gather_statistics, skip_operations);
}

void ReplayInterpreter::buildOperations()
{
  if (! m_ops->compileNeeded() )
    return;

  ReplayFile::Header* hdr = m_ops->getOperationsTable();

  hdr->num_allocators = 0;
  ReplayFile::Operation* op = &hdr->ops[0];
  memset(op, 0, sizeof(*op));
  op->op_type = ReplayFile::otype::ALLOCATE;
  hdr->num_operations = 1;

  // Get the input file size
  m_input_file.seekg(0, std::ios::end);
  auto filesize = m_input_file.tellg();
  m_input_file.seekg(0, std::ios::beg);
  int percent_complete = 0;

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
    else if ( m_json["event"] == "copy" ) {
      replay_compileCopy();
    }
    else if ( m_json["event"] == "reallocate_ex" ) {
      replay_compileReallocate_ex();
    }
    else if ( m_json["event"] == "reallocate" ) {
      replay_compileReallocate();
    }
    else if ( m_json["event"] == "setDefaultAllocator" ) {
      replay_compileSetDefaultAllocator();
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

    //
    // Report progress in parsing file
    //
    auto current_pos = m_input_file.tellg();
    double numerator = static_cast<double>(current_pos);
    double denominator = static_cast<double>(filesize);
    double percentage = (numerator / denominator) * 100.0;
    int wholepercentage = percentage;

    if (wholepercentage != percent_complete) {
      percent_complete = wholepercentage;
      std::cout << percent_complete << "%\r" << std::flush;
    }
  }
  std::cout << percent_complete << "%100 - Compilation complete" << std::endl;

  //
  // Flush operations to compile file and read back in read-only (PRIVATE) mode
  //
  delete m_ops;
  m_ops = new ReplayFile{m_input_file_name, m_input_file_name + ".bin"};
}

void ReplayInterpreter::printAllocators(ReplayFile* rf)
{
  auto optable = rf->getOperationsTable();
  std::cerr << rf->m_input_filename << std::endl;
  for (std::size_t i{0}; i < optable->num_allocators; ++i) {
    switch (optable->allocators[i].type) {
      default: std::cerr << "?? "; break;
      case ReplayFile::MEMORY_RESOURCE: std::cerr << " MEMORY_RESOURCE "; break;
      case ReplayFile::ALLOCATION_ADVISOR: std::cerr << " ALLOCATION_ADVISOR "; break;
      case ReplayFile::DYNAMIC_POOL_LIST: std::cerr << " DYNAMIC_POOL_LIST "; break;
      case ReplayFile::DYNAMIC_POOL_MAP: std::cerr << " DYNAMIC_POOL_MAP "; break;
      case ReplayFile::MONOTONIC: std::cerr << " MONOTONIC "; break;
      case ReplayFile::SLOT_POOL: std::cerr << " SLOT_POOL "; break;
      case ReplayFile::SIZE_LIMITER: std::cerr << " SIZE_LIMITER "; break;
      case ReplayFile::THREADSAFE_ALLOCATOR: std::cerr << " THREADSAFE_ALLOCATOR "; break;
      case ReplayFile::FIXED_POOL: std::cerr << " FIXED_POOL "; break;
      case ReplayFile::MIXED_POOL: std::cerr << " MIXED_POOL "; break;
      case ReplayFile::ALLOCATION_PREFETCHER: std::cerr << " ALLOCATION_PREFETCHER "; break;
      case ReplayFile::NUMA_POLICY: std::cerr << " NUMA_POLICY "; break;
    }

    std::cerr
      << optable->allocators[i].base_name << ", "
      << optable->allocators[i].name << std::endl;
  }
  std::cerr << std::endl;
}

bool ReplayInterpreter::compareOperations(ReplayInterpreter& rh)
{
  bool rval = true;

  if (m_ops->getOperationsTable()->m.version != rh.m_ops->getOperationsTable()->m.version) {
    std::cerr << "Number of version mismatch: "
        << m_ops->getOperationsTable()->m.version
        << " != " << rh.m_ops->getOperationsTable()->m.version
        << std::endl;
    rval = false;
  }

  if (m_ops->getOperationsTable()->num_allocators != rh.m_ops->getOperationsTable()->num_allocators) {
    std::cerr << "Number of allocators mismatch: "
        << m_ops->getOperationsTable()->num_allocators
        << " != " << rh.m_ops->getOperationsTable()->num_allocators
        << std::endl;
    printAllocators(m_ops);
    printAllocators(rh.m_ops);
    rval = false;
  }

  if (m_ops->getOperationsTable()->num_operations != rh.m_ops->getOperationsTable()->num_operations) {
    std::cerr << "Number of operations mismatch: "
        << m_ops->getOperationsTable()->num_operations
        << " != " << rh.m_ops->getOperationsTable()->num_operations
        << std::endl;
    rval = false;
  }

  if ( rval != false ) {
    for ( std::size_t i = 0; i < rh.m_ops->getOperationsTable()->num_allocators; ++i) {
      if (bcmp(   &m_ops->getOperationsTable()->allocators[i]
                , &rh.m_ops->getOperationsTable()->allocators[i]
                , sizeof(ReplayFile::AllocatorTableEntry)))
      {
        std::cerr << "AllocatorTable data miscompare at index " << i
          << ", sizeof Entry is: " << sizeof(ReplayFile::AllocatorTableEntry)
          << std::endl;

        unsigned char* lhs = (unsigned char*)(&m_ops->getOperationsTable()->allocators[i]);
        unsigned char* rhs = (unsigned char*)(&rh.m_ops->getOperationsTable()->allocators[i]);

        for ( std::size_t j = 0; j < sizeof(ReplayFile::AllocatorTableEntry); ++j ) {
          if (lhs[j] != rhs[j]) {
            std::cerr << j << ": " << (unsigned int)lhs[j] << " ~= " << (unsigned int)rhs[j] << std::endl;
          }
        }

        rval = false;
      }
    }

    for (std::size_t i = 0; i < rh.m_ops->getOperationsTable()->num_operations; ++i) {
      if (bcmp(   &m_ops->getOperationsTable()->ops[i]
                , &rh.m_ops->getOperationsTable()->ops[i]
                , sizeof(ReplayFile::Operation)))
      {
        std::cerr << "Operations table data miscompare at operation #" << i << std::endl;
        std::cerr << "    LHS: "
          << "type=" << m_ops->getOperationsTable()->ops[i].op_type
          << ", op_allocator=" << m_ops->getOperationsTable()->ops[i].op_allocator
          << ", op_alloc_ops[1]=" << m_ops->getOperationsTable()->ops[i].op_alloc_ops[1]
          << ", op_alloc_ops[0]=" << m_ops->getOperationsTable()->ops[i].op_alloc_ops[0]
          << ", size=" << m_ops->getOperationsTable()->ops[i].op_size
          << ", op_allocated_ptr=" << m_ops->getOperationsTable()->ops[i].op_allocated_ptr
          << std::endl;
        std::cerr << "    RHS: "
          << "type=" << rh.m_ops->getOperationsTable()->ops[i].op_type
          << ", op_allocator=" << rh.m_ops->getOperationsTable()->ops[i].op_allocator
          << ", op_alloc_ops[1]=" << rh.m_ops->getOperationsTable()->ops[i].op_alloc_ops[1]
          << ", op_alloc_ops[0]=" << rh.m_ops->getOperationsTable()->ops[i].op_alloc_ops[0]
          << ", size=" << rh.m_ops->getOperationsTable()->ops[i].op_size
          << ", op_allocated_ptr=" << rh.m_ops->getOperationsTable()->ops[i].op_allocated_ptr
          << std::endl;

        rval = false;
      }
    }
  }

  return rval;
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
  const uint64_t obj_p {
      getPointer( std::string{m_json["result"]} )
  };
  ReplayFile::Header* hdr = m_ops->getOperationsTable();

  m_allocator_indices[obj_p] = hdr->num_allocators;

  ReplayFile::AllocatorTableEntry* alloc = &(hdr->allocators[hdr->num_allocators]);

  alloc->type = ReplayFile::rtype::MEMORY_RESOURCE;
  alloc->introspection = false;
  alloc->argc = 0;
  m_ops->copyString(allocator_name, alloc->name);

  ReplayFile::Operation* op = &hdr->ops[hdr->num_operations];
  memset(op, 0, sizeof(*op));

  op->op_type = ReplayFile::otype::ALLOCATOR_CREATION;
  op->op_allocator = hdr->num_allocators;
  m_allocator_index[allocator_name] = hdr->num_allocators;
  hdr->num_allocators++;
  hdr->num_operations++;
}

void ReplayInterpreter::replay_compileAllocator( void )
{
  ReplayFile::Header* hdr = m_ops->getOperationsTable();

  ReplayFile::AllocatorTableEntry* alloc =
            & (m_ops->getOperationsTable()->allocators[hdr->num_allocators]);

  const std::string allocator_name{m_json["payload"]["allocator_name"]};

  if ( m_json["result"].is_null() ) {
    const bool introspection{m_json["payload"]["with_introspection"]};
    const std::string raw_mangled_type{m_json["payload"]["type"]};

    m_ops->copyString(allocator_name, alloc->name);
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
      m_ops->copyString(base_allocator_name, alloc->base_name);
      m_ops->copyString(advice_operation, alloc->argv.advisor.advice);
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
          m_ops->copyString(accessing_allocator_name, alloc->argv.advisor.accessing_allocator);
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
          m_ops->copyString(accessing_allocator_name, alloc->argv.advisor.accessing_allocator);
          break;
        }
      }
    }
    else if ( type == "umpire::strategy::AllocationPrefetcher" ) {
      const std::string base_allocator_name{m_json["payload"]["args"][0]};

      alloc->type = ReplayFile::rtype::ALLOCATION_PREFETCHER;
      m_ops->copyString(base_allocator_name, alloc->base_name);
    }
    else if ( type == "umpire::strategy::NumaPolicy" ) {
      const std::string base_allocator_name{m_json["payload"]["args"][0]};

      alloc->type = ReplayFile::rtype::NUMA_POLICY;
      get_from_string(m_json["payload"]["args"][1], alloc->argv.numa.node);

      m_ops->copyString(base_allocator_name, alloc->base_name);
    }
    else if ( type == "umpire::strategy::QuickPool" ) {
      const std::string base_allocator_name{m_json["payload"]["args"][0]};

      alloc->type = ReplayFile::rtype::QUICKPOOL;

      m_ops->copyString(base_allocator_name, alloc->base_name);

      // Now grab the optional fields
      if (alloc->argc >= 4) {
        alloc->argc = 4;    // strip heuristic parameter
        get_from_string(m_json["payload"]["args"][1],
                        alloc->argv.pool.initial_alloc_size);
        get_from_string(m_json["payload"]["args"][2],
                        alloc->argv.pool.min_alloc_size);
        get_from_string(m_json["payload"]["args"][3],
                        alloc->argv.pool.alignment);
      }
      else if (alloc->argc >= 3) {
        alloc->argc = 3;    // strip heuristic parameter
        get_from_string(m_json["payload"]["args"][1], alloc->argv.pool.initial_alloc_size);
        get_from_string(m_json["payload"]["args"][2], alloc->argv.pool.min_alloc_size);
      }
      else if (alloc->argc == 2) {
        get_from_string(m_json["payload"]["args"][1],
            alloc->argv.pool.initial_alloc_size);
      }
    }
    else if ( type == "umpire::strategy::DynamicPoolList" ) {
      const std::string base_allocator_name{m_json["payload"]["args"][0]};

      alloc->type = ReplayFile::rtype::DYNAMIC_POOL_LIST;

      m_ops->copyString(base_allocator_name, alloc->base_name);

      // Now grab the optional fields
      if (alloc->argc >= 4) {
        alloc->argc = 4;    // strip heuristic parameter
        get_from_string(m_json["payload"]["args"][1],
                        alloc->argv.dynamic_pool_list.initial_alloc_size);
        get_from_string(m_json["payload"]["args"][2],
                        alloc->argv.dynamic_pool_list.min_alloc_size);
        get_from_string(m_json["payload"]["args"][3],
                        alloc->argv.dynamic_pool_list.alignment);
      }
      else if (alloc->argc >= 3) {
        alloc->argc = 3;    // strip heuristic parameter
        get_from_string(m_json["payload"]["args"][1], alloc->argv.dynamic_pool_list.initial_alloc_size);
        get_from_string(m_json["payload"]["args"][2], alloc->argv.dynamic_pool_list.min_alloc_size);
      }
      else if (alloc->argc == 2) {
        get_from_string(m_json["payload"]["args"][1],
            alloc->argv.dynamic_pool_list.initial_alloc_size);
      }
    }
    else if ( type == "umpire::strategy::DynamicPool" || type == "umpire::strategy::DynamicPoolMap" ) {
      const std::string base_allocator_name{m_json["payload"]["args"][0]};

      alloc->type = ReplayFile::rtype::DYNAMIC_POOL_MAP;
      m_ops->copyString(base_allocator_name, alloc->base_name);

      if (alloc->argc >= 4) {
        alloc->argc = 4;    // strip heuristic parameter
        get_from_string(m_json["payload"]["args"][1],
                        alloc->argv.dynamic_pool_map.initial_alloc_size);
        get_from_string(m_json["payload"]["args"][2],
                        alloc->argv.dynamic_pool_map.min_alloc_size);
        get_from_string(m_json["payload"]["args"][3],
                        alloc->argv.dynamic_pool_map.alignment);
      }
      else if (alloc->argc >= 3) {
        alloc->argc = 3;    // strip heuristic parameter
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
      m_ops->copyString(base_allocator_name, alloc->base_name);

      get_from_string(m_json["payload"]["args"][1],
                      alloc->argv.monotonic_pool.capacity);
    }
    else if ( type == "umpire::strategy::SlotPool" ) {
      const std::string base_allocator_name{m_json["payload"]["args"][0]};

      alloc->type = ReplayFile::rtype::SLOT_POOL;
      m_ops->copyString(base_allocator_name, alloc->base_name);
      get_from_string(m_json["payload"]["args"][1], alloc->argv.slot_pool.slots);
    }
    else if ( type == "umpire::strategy::SizeLimiter" ) {
      const std::string base_allocator_name{m_json["payload"]["args"][0]};

      alloc->type = ReplayFile::rtype::SIZE_LIMITER;
      m_ops->copyString(base_allocator_name, alloc->base_name);
      get_from_string(m_json["payload"]["args"][1], alloc->argv.size_limiter.size_limit);
    }
    else if ( type == "umpire::strategy::ThreadSafeAllocator" ) {
      const std::string base_allocator_name{m_json["payload"]["args"][0]};

      alloc->type = ReplayFile::rtype::THREADSAFE_ALLOCATOR;
      m_ops->copyString(base_allocator_name, alloc->base_name);
    }
    else if ( type == "umpire::strategy::FixedPool" ) {
      const std::string base_allocator_name{m_json["payload"]["args"][0]};

      alloc->type = ReplayFile::rtype::FIXED_POOL;
      m_ops->copyString(base_allocator_name, alloc->base_name);
      get_from_string(m_json["payload"]["args"][1], alloc->argv.fixed_pool.object_bytes);

      // Now grab the optional fields
      if (alloc->argc == 3) {
        get_from_string(m_json["payload"]["args"][2], alloc->argv.fixed_pool.objects_per_pool);
      }
    }
    else if ( type == "umpire::strategy::MixedPool" ) {
      const std::string base_allocator_name{m_json["payload"]["args"][0]};

      alloc->type = ReplayFile::rtype::MIXED_POOL;
      m_ops->copyString(base_allocator_name, alloc->base_name);

      // Now grab the optional fields
      if (alloc->argc >= 8) {
        alloc->argc = 8;    // strip heuristic parameter
        get_from_string(m_json["payload"]["args"][1], alloc->argv.mixed_pool.smallest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][2], alloc->argv.mixed_pool.largest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][3], alloc->argv.mixed_pool.max_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][4], alloc->argv.mixed_pool.size_multiplier);
        get_from_string(m_json["payload"]["args"][5], alloc->argv.mixed_pool.dynamic_initial_alloc_bytes);
        get_from_string(m_json["payload"]["args"][6], alloc->argv.mixed_pool.dynamic_min_alloc_bytes);
        get_from_string(m_json["payload"]["args"][7], alloc->argv.mixed_pool.dynamic_align_bytes);
      }
      else if (alloc->argc >= 7) {
        alloc->argc = 7;    // strip heuristic parameter
        get_from_string(m_json["payload"]["args"][1], alloc->argv.mixed_pool.smallest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][2], alloc->argv.mixed_pool.largest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][3], alloc->argv.mixed_pool.max_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][4], alloc->argv.mixed_pool.size_multiplier);
        get_from_string(m_json["payload"]["args"][5], alloc->argv.mixed_pool.dynamic_initial_alloc_bytes);
        get_from_string(m_json["payload"]["args"][6], alloc->argv.mixed_pool.dynamic_min_alloc_bytes);
      }
      else if (alloc->argc >= 6) {
        alloc->argc = 6;    // strip heuristic parameter
        get_from_string(m_json["payload"]["args"][1], alloc->argv.mixed_pool.smallest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][2], alloc->argv.mixed_pool.largest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][3], alloc->argv.mixed_pool.max_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][4], alloc->argv.mixed_pool.size_multiplier);
        get_from_string(m_json["payload"]["args"][5], alloc->argv.mixed_pool.dynamic_initial_alloc_bytes);
      }
      else if (alloc->argc >= 5) {
        alloc->argc = 5;    // strip heuristic parameter
        get_from_string(m_json["payload"]["args"][1], alloc->argv.mixed_pool.smallest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][2], alloc->argv.mixed_pool.largest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][3], alloc->argv.mixed_pool.max_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][4], alloc->argv.mixed_pool.size_multiplier);
      }
      else if (alloc->argc >= 4) {
        alloc->argc = 4;    // strip heuristic parameter
        get_from_string(m_json["payload"]["args"][1], alloc->argv.mixed_pool.smallest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][2], alloc->argv.mixed_pool.largest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][3], alloc->argv.mixed_pool.max_fixed_blocksize);
      }
      else if (alloc->argc >= 3) {
        alloc->argc = 3;    // strip heuristic parameter
        get_from_string(m_json["payload"]["args"][1], alloc->argv.mixed_pool.smallest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][2], alloc->argv.mixed_pool.largest_fixed_blocksize);
      }
      else if (alloc->argc >= 2) {
        alloc->argc = 2;    // strip heuristic parameter
        get_from_string(m_json["payload"]["args"][1], alloc->argv.mixed_pool.smallest_fixed_blocksize);
      }
    }
    else {
      REPLAY_ERROR("Unknown class (" << type << "), skipping.");
    }
  }
  else {
    const uint64_t obj_p{ getPointer(std::string{m_json["result"]["allocator_ref"]}) };

    m_allocator_indices[obj_p] = hdr->num_allocators;

    ReplayFile::Operation* op = &hdr->ops[hdr->num_operations];
    memset(op, 0, sizeof(*op));

    op->op_type = ReplayFile::otype::ALLOCATOR_CREATION;
    op->op_allocator = hdr->num_allocators;

    m_allocator_index[allocator_name] = hdr->num_allocators;

    hdr->num_allocators++;
    hdr->num_operations++;
  }
}

void ReplayInterpreter::replay_compileAllocate( void )
{
  if (m_replaying_reallocate)
    return;

  ReplayFile::Header* hdr = m_ops->getOperationsTable();
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
    memset(op, 0, sizeof(*op));

    op->op_type = ReplayFile::otype::ALLOCATE;
    op->op_allocator = getAllocatorIndex(std::string{m_json["payload"]["allocator_ref"]});
    op->op_size = alloc_size;
  }
  else {
    const uint64_t memory_ptr{
      getPointer( std::string{m_json["result"]["memory_ptr"]} )
    };

    m_allocation_id[memory_ptr] = hdr->num_operations;
    hdr->num_operations++;
  }
}

void ReplayInterpreter::replay_compileSetDefaultAllocator( void )
{
  ReplayFile::Header* hdr = m_ops->getOperationsTable();
  ReplayFile::Operation* op = &hdr->ops[hdr->num_operations];

  memset(op, 0, sizeof(*op));
  op->op_type = ReplayFile::otype::SETDEFAULTALLOCATOR;
  op->op_allocator = getAllocatorIndex(std::string{m_json["payload"]["allocator_ref"]});
  hdr->num_operations++;
}

int ReplayInterpreter::getAllocatorIndex(std::string ref_s)
{
  const uint64_t ref_p{std::stoul(ref_s, nullptr, 0)};
  auto n_iter(m_allocator_indices.find(ref_p));

  if ( n_iter == m_allocator_indices.end() )
    REPLAY_ERROR("Unable to find allocator: " << ref_s);

  return n_iter->second;
}

uint64_t ReplayInterpreter::getPointer(std::string ptr_name)
{
  const uint64_t ptr{std::stoul(ptr_name, nullptr, 0)};

  return ptr;
}

void ReplayInterpreter::replay_compileCopy( void )
{
  if (m_replaying_reallocate)
    return;

  ReplayFile::Header* hdr = m_ops->getOperationsTable();
  ReplayFile::Operation* op = &hdr->ops[hdr->num_operations++];
  memset(op, 0, sizeof(*op));

  uint64_t src_ptr{ getPointer(std::string{m_json["payload"]["src"]}) };
  uint64_t dst_ptr{ getPointer(std::string{m_json["payload"]["dest"]}) };
  uint64_t src_off{m_json["payload"]["src_offset"]};
  uint64_t dst_off{m_json["payload"]["dst_offset"]};

  src_ptr -= src_off;
  dst_ptr -= dst_off;

  op->op_type = ReplayFile::otype::COPY;
  op->op_size = m_json["payload"]["size"];
  op->op_offsets[0] = m_json["payload"]["src_offset"];
  op->op_offsets[1] = m_json["payload"]["dst_offset"];
  op->op_alloc_ops[0] = m_allocation_id[src_ptr];
  op->op_alloc_ops[1] = m_allocation_id[dst_ptr];
}

void ReplayInterpreter::replay_compileReallocate( void )
{
  const std::size_t alloc_size{m_json["payload"]["size"]};
  const uint64_t ptr{
    getPointer( std::string{m_json["payload"]["ptr"]} )
  };

  m_replaying_reallocate = true;

  ReplayFile::Header* hdr = m_ops->getOperationsTable();
  ReplayFile::Operation* op = &hdr->ops[hdr->num_operations];

  if ( m_json["result"].is_null() ) {
    memset(op, 0, sizeof(*op));
    op->op_type = ReplayFile::otype::REALLOCATE;
    op->op_alloc_ops[1] = (ptr == 0) ? 0 : m_allocation_id[ptr];
    op->op_size = alloc_size;
  }
  else {
    const uint64_t memory_ptr{
      getPointer( std::string{m_json["result"]["memory_ptr"]} )
    };

    m_allocation_id[memory_ptr] = hdr->num_operations;
    hdr->num_operations++;
    m_replaying_reallocate = false;
  }
}

void ReplayInterpreter::replay_compileReallocate_ex( void )
{
  const std::size_t alloc_size{m_json["payload"]["size"]};
  const uint64_t ptr{
      getPointer( std::string{m_json["payload"]["ptr"]} )
  };

  m_replaying_reallocate = true;

  ReplayFile::Header* hdr = m_ops->getOperationsTable();
  ReplayFile::Operation* op = &hdr->ops[hdr->num_operations];

  if ( m_json["result"].is_null() ) {
    memset(op, 0, sizeof(*op));
    op->op_type = ReplayFile::otype::REALLOCATE_EX;
    op->op_alloc_ops[1] = (ptr == 0) ? 0 : m_allocation_id[ptr];
    op->op_size = alloc_size;
    op->op_allocator = getAllocatorIndex(std::string{m_json["payload"]["allocator_ref"]});
  }
  else {
    const std::string memory_str{m_json["result"]["memory_ptr"]};
    const uint64_t memory_ptr{
      getPointer( std::string{m_json["result"]["memory_ptr"]} )
    };

    m_allocation_id[memory_ptr] = hdr->num_operations;
    hdr->num_operations++;
    m_replaying_reallocate = false;
  }
}

void ReplayInterpreter::replay_compileDeallocate( void )
{
  if (m_replaying_reallocate)
    return;

  ReplayFile::Header* hdr = m_ops->getOperationsTable();
  ReplayFile::Operation* op = &hdr->ops[hdr->num_operations];
  memset(op, 0, sizeof(*op));

  op->op_type = ReplayFile::otype::DEALLOCATE;
  op->op_allocator = getAllocatorIndex(std::string{m_json["payload"]["allocator_ref"]});

  const uint64_t memory_ptr {
      getPointer( std::string{m_json["payload"]["memory_ptr"]} )
  };

  op->op_alloc_ops[0] = m_allocation_id[memory_ptr];
  hdr->num_operations++;
}

void ReplayInterpreter::replay_compileCoalesce( void )
{
  std::string allocator_name{m_json["payload"]["allocator_name"]};
  strip_off_base(allocator_name);

  ReplayFile::Header* hdr = m_ops->getOperationsTable();
  ReplayFile::Operation* op = &hdr->ops[hdr->num_operations];

  memset(op, 0, sizeof(*op));
  op->op_type = ReplayFile::otype::COALESCE;
  op->op_allocator = m_allocator_index[allocator_name];
  hdr->num_operations++;
}

void ReplayInterpreter::replay_compileRelease( void )
{
  ReplayFile::Header* hdr = m_ops->getOperationsTable();
  ReplayFile::Operation* op = &hdr->ops[hdr->num_operations];

  memset(op, 0, sizeof(*op));
  op->op_type = ReplayFile::otype::RELEASE;
  op->op_allocator = getAllocatorIndex(std::string{m_json["payload"]["allocator_ref"]});
  hdr->num_operations++;
}
#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)

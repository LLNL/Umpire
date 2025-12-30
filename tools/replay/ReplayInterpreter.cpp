//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#if !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
#include <cxxabi.h>   // for __cxa_demangle

#include "umpire/config.hpp"

#include "ReplayFile.hpp"
#include "ReplayInterpreter.hpp"
#include "ReplayMacros.hpp"
#include "ReplayOperationManager.hpp"
#include "ReplayOptions.hpp"
#include "umpire/event/event.hpp"
#include "umpire/json/json.hpp"

#if defined(UMPIRE_ENABLE_SQLITE_EXPERIMENTAL)
#include "umpire/event/sqlite_database.hpp"
#else
#include "umpire/event/json_file_store.hpp"
#endif

ReplayInterpreter::ReplayInterpreter( const ReplayOptions& options ) : m_options(options)
{
  m_ops = new ReplayFile{m_options};
}

ReplayInterpreter::~ReplayInterpreter()
{
  if (m_ops != nullptr) {
    delete m_ops;
    m_ops = nullptr;
  }
}

void ReplayInterpreter::runOperations()
{
  ReplayOperationManager m_operation_mgr{m_options, m_ops, m_ops->getOperationsTable()};

  m_operation_mgr.runOperations();
}

void ReplayInterpreter::buildOperations()
{
  ReplayFile::Header* hdr{nullptr};
  ReplayFile::Operation* op{nullptr};

  if ( ! m_ops->compileNeeded() ) {
    return;
  }

  hdr = m_ops->getOperationsTable();
  hdr->num_allocators = 0;
  op = &hdr->ops[0];
  memset(op, 0, sizeof(*op));
  op->op_type = ReplayFile::otype::ALLOCATE;
  op->op_line_number = m_line_number;
  hdr->num_operations = 1;

#if defined(UMPIRE_ENABLE_SQLITE_EXPERIMENTAL)
umpire::event::sqlite_database store{m_options.input_file};
#else
umpire::event::json_file_store store{m_options.input_file, true};
#endif
  std::vector<umpire::event::event> events;
  events = store.get_events();

  for (auto e : events) {
    m_line_number++;

    m_event = e;

    if ( m_event.cat != umpire::event::category::operation && m_event.name == "version" ) {
      m_version_ops++;
      m_log_version_major = m_event.numeric_args["major"];
      m_log_version_minor = m_event.numeric_args["minor"];
      m_log_version_patch = m_event.numeric_args["patch"];

      if (   m_log_version_major != UMPIRE_VERSION_MAJOR
          || m_log_version_minor != UMPIRE_VERSION_MINOR
          || m_log_version_patch != UMPIRE_VERSION_PATCH ) {
        REPLAY_WARNING("Warning, version mismatch:\n"
          << "  Tool version: " << UMPIRE_VERSION_MAJOR << "."
          << UMPIRE_VERSION_MINOR << "." << UMPIRE_VERSION_PATCH << std::endl
          << "  Log  version: "
          << m_log_version_major << "."
          << m_log_version_minor  << "."
          << m_log_version_patch);

        if (m_log_version_major != UMPIRE_VERSION_MAJOR) {
          REPLAY_WARNING("Warning, major version mismatch - attempting replay anyway...\n"
            << "  Tool version: " << UMPIRE_VERSION_MAJOR << "."
            << UMPIRE_VERSION_MINOR << "." << UMPIRE_VERSION_PATCH << std::endl
            << "  Log  version: "
            << m_log_version_major << "."
            << m_log_version_minor  << "."
            << m_log_version_patch);
        }
      }
      continue;
    }

    if ( m_event.cat != umpire::event::category::operation)
      continue;

    try {
      if ( m_event.name == "allocate" ) {
        m_allocate_ops++;
        compile_allocate();
      }
      else if ( m_event.name == "deallocate" ) {
        m_deallocate_ops++;
        if (!compile_deallocate()) {
          continue;
        }
      }
      else if ( m_event.name == "make_allocator" ) {
        m_make_allocator_ops++;
        compile_make_allocator();
      }
      else if ( m_event.name == "make_memory_resource" ) {
        m_make_memory_resource_ops++;
        compile_make_memory_resource();
      }
      else if ( m_event.name == "copy" ) {
        m_copy_ops++;
      }
      else if ( m_event.name == "move" ) {
        m_move_ops++;
      }
      else if ( m_event.name == "reallocate" ) {
        m_reallocate_ops++;
        compile_reallocate();
      }
      else if ( m_event.name == "set_default_allocator" ) {
        m_set_default_allocator_ops++;
        compile_set_default_allocator();
      }
      else if ( m_event.name == "coalesce" ) {
        m_coalesce_ops++;
        compile_coalesce();
      }
      else if ( m_event.name == "release" ) {
        m_release_ops++;
        compile_release();
      }
      else if ( m_event.name == "register_external_allocation" ) {
        m_register_external_pointer++;
      }
      else if ( m_event.name == "deregister_external_allocation" ) {
        m_deregister_external_pointer++;
      }
      else {
        REPLAY_ERROR("Unknown Replay Operation: " << m_ops->getLine(m_line_number));
      }
    }
    catch (...) {
      REPLAY_ERROR("Failed to compile: " << m_ops->getLine(m_line_number));
    }
  }

  //
  // Flush operations to compile file and read back in read-only (PRIVATE) mode
  //
  delete m_ops;
  m_ops = new ReplayFile{m_options};

  if ( ! m_options.quiet ) {
    const std::size_t leaked_allocations{m_allocate_ops - m_deallocate_ops};

    std::cout
      << "Replay File Version: " << m_log_version_major << "." << m_log_version_minor << "." << m_log_version_patch << std::endl
      << std::setw(12) << m_make_memory_resource_ops << " makeMemoryResource operations" << std::endl
      << std::setw(12) << m_make_allocator_ops << " makeAllocator operations" << std::endl
      << std::endl
      << std::setw(12) << m_allocate_ops << " allocate operations" << std::endl
      << std::setw(12) << m_deallocate_ops << " deallocate performed (" << leaked_allocations << " leaked)" << std::endl;

    if (m_deallocate_rogue_ignored) {
      std::cout << "    " << std::setw(12) << m_deallocate_rogue_ignored << " skipped due to being rogue" << std::endl;
    }
    
    std::cout << std::endl
      << std::setw(12) << m_register_external_pointer << " external registrations (not replayed)" << std::endl
      << std::setw(12) << m_deregister_external_pointer << " external deregistrations (not replayed)" << std::endl
      << std::endl
      << std::setw(12) << m_copy_ops << " copy operations" << std::endl
      << std::setw(12) << m_move_ops << " move operations" << std::endl
      << std::setw(12) << m_reallocate_ops << " reallocate operations" << std::endl
      << std::setw(12) << m_set_default_allocator_ops << " setDefaultAllocator operations" << std::endl
      << std::setw(12) << m_coalesce_ops << " coalesce operations" << std::endl
      << std::setw(12) << m_release_ops << " release operations" << std::endl
      << std::setw(12) << m_version_ops << " version operations"
      << std::endl;
  }
}

std::string ReplayInterpreter::printAllocatorInfo(ReplayFile::AllocatorTableEntry* allocator)
{
  std::stringstream ss;

  ss << "Line#: " << allocator->line_number 
    << ", argc: " << allocator->argc
    << ", Name: " << allocator->name
    << ", Basename: " << allocator->base_name
    << ", Type: ";

  switch (allocator->type) {
    default: ss << "??"; break;
    case ReplayFile::MEMORY_RESOURCE: ss << " MEMORY_RESOURCE"; break;
    case ReplayFile::ALLOCATION_ADVISOR: ss << " ALLOCATION_ADVISOR"; break;
    case ReplayFile::DYNAMIC_POOL_LIST: ss << " DYNAMIC_POOL_LIST"; break;
    case ReplayFile::QUICKPOOL: ss << " QUICKPOOL"; break;
    case ReplayFile::MONOTONIC: ss << " MONOTONIC"; break;
    case ReplayFile::SLOT_POOL: ss << " SLOT_POOL"; break;
    case ReplayFile::SIZE_LIMITER: ss << " SIZE_LIMITER"; break;
    case ReplayFile::THREADSAFE_ALLOCATOR: ss << " THREADSAFE_ALLOCATOR"; break;
    case ReplayFile::FIXED_POOL: ss << " FIXED_POOL"; break;
    case ReplayFile::MIXED_POOL: ss << " MIXED_POOL"; break;
    case ReplayFile::ALLOCATION_PREFETCHER: ss << " ALLOCATION_PREFETCHER"; break;
    case ReplayFile::NUMA_POLICY: ss << " NUMA_POLICY"; break;
  }
  return ss.str();
}

void ReplayInterpreter::printAllocators(ReplayFile* rf)
{
  auto optable = rf->getOperationsTable();
  std::cerr << rf->getInputFileName() << std::endl;
  for (std::size_t i{0}; i < optable->num_allocators; ++i) {
    std::cerr << printAllocatorInfo(&(optable->allocators[i])) << std::endl;
  }
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
  }

  if ( rval != false ) {
    for ( std::size_t i = 0; i < rh.m_ops->getOperationsTable()->num_allocators; ++i) {
      if ( m_ops->getOperationsTable()->allocators[i].type != rh.m_ops->getOperationsTable()->allocators[i].type ) {
        std::cerr << "AllocatorTable type data miscompare at type " << i << std::endl;
        rval = false;
      }

      if ( m_ops->getOperationsTable()->allocators[i].introspection != rh.m_ops->getOperationsTable()->allocators[i].introspection ) {
        std::cerr << "AllocatorTable introspection data miscompare at index " << i << std::endl;
        rval = false;
      }

      if ( strcmp( m_ops->getOperationsTable()->allocators[i].name, rh.m_ops->getOperationsTable()->allocators[i].name ) ) {
        std::cerr << "AllocatorTable name data miscompare at index " << i << std::endl;
        rval = false;
      }

      if ( strcmp( m_ops->getOperationsTable()->allocators[i].base_name, rh.m_ops->getOperationsTable()->allocators[i].base_name ) )   {
        std::cerr << "AllocatorTable base_name data miscompare at index " << i << std::endl;
        rval = false;
      }

      if ( m_ops->getOperationsTable()->allocators[i].argc != rh.m_ops->getOperationsTable()->allocators[i].argc ) {
        std::cerr << "AllocatorTable argc data miscompare at index " << i << std::endl
          << "    LHS: " << printAllocatorInfo(&m_ops->getOperationsTable()->allocators[i]) << std::endl
          << "    RHS: " << printAllocatorInfo(&rh.m_ops->getOperationsTable()->allocators[i]) << std::endl
          << std::endl;
        rval = false;
      }

      if (bcmp(   &m_ops->getOperationsTable()->allocators[i].argv
                , &rh.m_ops->getOperationsTable()->allocators[i].argv
                , sizeof(ReplayFile::AllocatorTableEntry::argv)))
      {
        std::cerr << "AllocatorTable Union data miscompare at index " << i << std::endl;
        rval = false;
      }
    }

    bool mismatch{false};
    for (std::size_t i = 0; i < rh.m_ops->getOperationsTable()->num_operations; ++i) {
      if ( m_ops->getOperationsTable()->ops[i].op_type != rh.m_ops->getOperationsTable()->ops[i].op_type ) {
        std::cerr << "Operation Type mismatch" << std::endl;
        mismatch = true;
      }
      if ( m_ops->getOperationsTable()->ops[i].op_allocator != rh.m_ops->getOperationsTable()->ops[i].op_allocator ) {
        std::cerr << "Operation Allocator mismatch" << std::endl;
        mismatch = true;
      }
      if ( m_ops->getOperationsTable()->ops[i].op_size != rh.m_ops->getOperationsTable()->ops[i].op_size ) {
        std::cerr << "Operation Size mismatch" << std::endl;
        mismatch = true;
      }
      if ( m_ops->getOperationsTable()->ops[i].op_offsets[0] != rh.m_ops->getOperationsTable()->ops[i].op_offsets[0] ) {
        std::cerr << "Operation Offset[0] mismatch" << std::endl;
        mismatch = true;
      }
      if ( m_ops->getOperationsTable()->ops[i].op_offsets[1] != rh.m_ops->getOperationsTable()->ops[i].op_offsets[1] ) {
        std::cerr << "Operation Offset[1] mismatch" << std::endl;
        mismatch = true;
      }
      if ( m_ops->getOperationsTable()->ops[i].op_alloc_ops[0] != rh.m_ops->getOperationsTable()->ops[i].op_alloc_ops[0] ) {
        std::cerr << "Operation Offset[0] mismatch" << std::endl;
        mismatch = true;
      }
      if ( m_ops->getOperationsTable()->ops[i].op_alloc_ops[1] != rh.m_ops->getOperationsTable()->ops[i].op_alloc_ops[1] ) {
        std::cerr << "Operation Offset[1] mismatch" << std::endl;
        mismatch = true;
      }
      if ( mismatch ) {
        std::cerr << "    LHS: "
          << m_ops->getLine(m_ops->getOperationsTable()->ops[i].op_line_number)
          << std::endl;
        std::cerr << "    RHS: "
          << rh.m_ops->getLine(rh.m_ops->getOperationsTable()->ops[i].op_line_number)
          << std::endl;
        rval = false;
        break;
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

void ReplayInterpreter::compile_make_memory_resource()
{
  const std::string allocator_name{ m_event.tags["allocator_name"] };
  const uint64_t obj_p { getPointer( m_event.string_args["allocator_ref"] ) };
  ReplayFile::Header* hdr = m_ops->getOperationsTable();

  m_allocator_indices[obj_p] = hdr->num_allocators;

  ReplayFile::AllocatorTableEntry* alloc = &(hdr->allocators[hdr->num_allocators]);

  alloc->type = ReplayFile::rtype::MEMORY_RESOURCE;
  alloc->line_number = m_line_number;
  alloc->introspection = true;
  alloc->argc = 0;
  m_ops->copyString(allocator_name, alloc->name);

  ReplayFile::Operation* op = &hdr->ops[hdr->num_operations];
  memset(op, 0, sizeof(*op));

  op->op_type = ReplayFile::otype::ALLOCATOR_CREATION;
  op->op_line_number = m_line_number;
  op->op_allocator = hdr->num_allocators;
  m_allocator_index[allocator_name] = hdr->num_allocators;
  hdr->num_allocators++;
  if (hdr->num_allocators >= ReplayFile::max_allocators) {
    REPLAY_ERROR("Too many allocators for replay: " << hdr->num_allocators);
  }

  hdr->num_operations++;
}

void ReplayInterpreter::compile_make_allocator()
{
  const std::string allocator_name{m_event.tags["allocator_name"]};
  const bool introspection{m_event.numeric_args["introspection"] == 1};
  const std::string raw_mangled_type{m_event.string_args["type"]};

  ReplayFile::Header* hdr{m_ops->getOperationsTable()};
  ReplayFile::AllocatorTableEntry* alloc{ &(m_ops->getOperationsTable()->allocators[hdr->num_allocators]) };

  alloc->line_number = m_line_number;

  m_ops->copyString(allocator_name, alloc->name);
  alloc->introspection = introspection;

  std::string type;
  if (!m_options.do_not_demangle && m_log_version_major >= 2) {
    const std::string type_prefix{raw_mangled_type.substr(0, 2)};

    // Add _Z so that we can demangle the external symbol
    const std::string mangled_type{ (type_prefix == "_Z") ? raw_mangled_type : std::string{"_Z"} + raw_mangled_type };

    auto result = abi::__cxa_demangle( mangled_type.c_str(), nullptr, nullptr, nullptr);
    if (!result) {
        REPLAY_ERROR("Failed to demangle strategy type. Mangled type: " << mangled_type);
    }
    type = std::string{result};
    ::free(result);
  } else {
    type = raw_mangled_type;
  }

  const std::string base_allocator_name{m_event.string_args["arg0"]};
  alloc->argc = 1;

  if ( type == "umpire::strategy::AllocationAdvisor" ) {
    const std::string advice_operation{m_event.string_args["arg1"]};
    alloc->argc++;

    int device_id{-1};   // Use default argument if negative

    if ( m_event.string_args.find("arg2") != m_event.string_args.end() ) { // Accessing Allocator provided
      alloc->argc++;

      if ( m_event.numeric_args.find("arg3") != m_event.numeric_args.end() ) { // ID provided as arg3
        alloc->argc++;
        device_id = m_event.numeric_args["arg3"];
      }
    }
    else {  // No accessing Allocator provided
      if ( m_event.numeric_args.find("arg2") != m_event.numeric_args.end() ) { // ID provided as arg2
        alloc->argc++;
        device_id = m_event.numeric_args["arg2"];
      }
    }

    alloc->type = ReplayFile::rtype::ALLOCATION_ADVISOR;
    m_ops->copyString(base_allocator_name, alloc->base_name);
    m_ops->copyString(advice_operation, alloc->argv.advisor.advice);
    alloc->argv.advisor.device_id = device_id;

    if (device_id >= 0) { // Optional device ID specified
      switch ( alloc->argc ) {
      default:
        REPLAY_ERROR("Invalid number of arguments (" << alloc->argc << " for " << type << " operation.  Stopping");
      case 3:
        break;
      case 4:
        const std::string accessing_allocator_name{m_event.string_args["arg2"]};
        m_ops->copyString(accessing_allocator_name, alloc->argv.advisor.accessing_allocator);
        break;
      }
    }
    else { // Use default device_id
      switch ( alloc->argc ) {
      default:
        REPLAY_ERROR("Invalid number of arguments (" << alloc->argc << " for " << type << " operation.  Stopping");
      case 2:
        break;
      case 3:
        const std::string accessing_allocator_name{m_event.string_args["arg2"]};
        m_ops->copyString(accessing_allocator_name, alloc->argv.advisor.accessing_allocator);
        break;
      }
    }
  }
  else if ( type == "umpire::strategy::AllocationPrefetcher" ) {
    alloc->type = ReplayFile::rtype::ALLOCATION_PREFETCHER;
    m_ops->copyString(base_allocator_name, alloc->base_name);
  }
  else if ( type == "umpire::strategy::NumaPolicy" ) {
    alloc->type = ReplayFile::rtype::NUMA_POLICY;
    alloc->argv.numa.node = m_event.numeric_args["arg1"];
    alloc->argc++;

    m_ops->copyString(base_allocator_name, alloc->base_name);
  }
  else if (type == "umpire::strategy::NamedAllocationStrategy") {
    alloc->type = ReplayFile::rtype::NAMED;
    m_ops->copyString(base_allocator_name, alloc->base_name);
  }
  else if ( type == "umpire::strategy::QuickPool" ) {
    alloc->type = ReplayFile::rtype::QUICKPOOL;
    m_ops->copyString(base_allocator_name, alloc->base_name);

    // Now grab the optional fields
    if (m_event.numeric_args.find("arg3") != m_event.numeric_args.end()) {
      alloc->argc = 4;    // ignore potential heuristic parameters
      alloc->argv.pool.initial_alloc_size = m_event.numeric_args["arg1"];
      alloc->argv.pool.min_alloc_size = m_event.numeric_args["arg2"];
      alloc->argv.pool.alignment = m_event.numeric_args["arg3"];
    }
    else if (m_event.numeric_args.find("arg2") != m_event.numeric_args.end()) {
      alloc->argc = 3;
      alloc->argv.pool.initial_alloc_size = m_event.numeric_args["arg1"];
      alloc->argv.pool.min_alloc_size = m_event.numeric_args["arg2"];
    }
    else if (m_event.numeric_args.find("arg1") != m_event.numeric_args.end()) {
      alloc->argc = 2;
      alloc->argv.pool.initial_alloc_size = m_event.numeric_args["arg1"];
    }
  }
  else if ( type == "umpire::strategy::DynamicPoolList" ) {
    alloc->type = ReplayFile::rtype::DYNAMIC_POOL_LIST;
    m_ops->copyString(base_allocator_name, alloc->base_name);

    // Now grab the optional fields
    if (m_event.numeric_args.find("arg3") != m_event.numeric_args.end()) {
      alloc->argc = 4;    // ignore potential heuristic parameters
      alloc->argv.pool.initial_alloc_size = m_event.numeric_args["arg1"];
      alloc->argv.pool.min_alloc_size = m_event.numeric_args["arg2"];
      alloc->argv.pool.alignment = m_event.numeric_args["arg3"];
    }
    else if (m_event.numeric_args.find("arg2") != m_event.numeric_args.end()) {
      alloc->argc = 3;
      alloc->argv.pool.initial_alloc_size = m_event.numeric_args["arg1"];
      alloc->argv.pool.min_alloc_size = m_event.numeric_args["arg2"];
    }
    else if (m_event.numeric_args.find("arg1") != m_event.numeric_args.end()) {
      alloc->argc = 2;
      alloc->argv.pool.initial_alloc_size = m_event.numeric_args["arg1"];
    }
  }
  else if ( type == "umpire::strategy::MonotonicAllocationStrategy" ) {
    alloc->type = ReplayFile::rtype::MONOTONIC;
    m_ops->copyString(base_allocator_name, alloc->base_name);
    alloc->argv.monotonic_pool.capacity = m_event.numeric_args["arg1"];
    alloc->argc++;
  }
  else if ( type == "umpire::strategy::SlotPool" ) {
    alloc->type = ReplayFile::rtype::SLOT_POOL;
    m_ops->copyString(base_allocator_name, alloc->base_name);
    alloc->argv.slot_pool.slots = m_event.numeric_args["arg1"];
    alloc->argc++;
  }
  else if ( type == "umpire::strategy::SizeLimiter" ) {
    alloc->type = ReplayFile::rtype::SIZE_LIMITER;
    m_ops->copyString(base_allocator_name, alloc->base_name);
    alloc->argv.size_limiter.size_limit = m_event.numeric_args["arg1"];
    alloc->argc++;
  }
  else if ( type == "umpire::strategy::ThreadSafeAllocator" ) {
    alloc->type = ReplayFile::rtype::THREADSAFE_ALLOCATOR;
    m_ops->copyString(base_allocator_name, alloc->base_name);
  }
  else if ( type == "umpire::strategy::FixedPool" ) {
    alloc->type = ReplayFile::rtype::FIXED_POOL;
    m_ops->copyString(base_allocator_name, alloc->base_name);
    alloc->argv.fixed_pool.object_bytes = m_event.numeric_args["arg1"];
    alloc->argc++;

    // Now grab the optional fields
    if (m_event.numeric_args.find("arg2") != m_event.numeric_args.end()) {
      alloc->argv.fixed_pool.objects_per_pool = m_event.numeric_args["arg2"];
      alloc->argc++;
    }
  }
  else if ( type == "umpire::strategy::MixedPool" ) {
    alloc->type = ReplayFile::rtype::MIXED_POOL;
    m_ops->copyString(base_allocator_name, alloc->base_name);

    // Now grab the optional fields
    if (m_event.numeric_args.find("arg7") != m_event.numeric_args.end()) {
      alloc->argc = 8;
      alloc->argv.mixed_pool.smallest_fixed_blocksize = m_event.numeric_args["arg1"];
      alloc->argv.mixed_pool.largest_fixed_blocksize = m_event.numeric_args["arg2"];
      alloc->argv.mixed_pool.max_fixed_blocksize = m_event.numeric_args["arg3"];
      alloc->argv.mixed_pool.size_multiplier = m_event.numeric_args["arg4"];
      alloc->argv.mixed_pool.dynamic_initial_alloc_bytes = m_event.numeric_args["arg5"];
      alloc->argv.mixed_pool.dynamic_min_alloc_bytes = m_event.numeric_args["arg6"];
      alloc->argv.mixed_pool.dynamic_align_bytes = m_event.numeric_args["arg7"];
    }
    else if (m_event.numeric_args.find("arg6") != m_event.numeric_args.end()) {
      alloc->argc = 7;
      alloc->argv.mixed_pool.smallest_fixed_blocksize = m_event.numeric_args["arg1"];
      alloc->argv.mixed_pool.largest_fixed_blocksize = m_event.numeric_args["arg2"];
      alloc->argv.mixed_pool.max_fixed_blocksize = m_event.numeric_args["arg3"];
      alloc->argv.mixed_pool.size_multiplier = m_event.numeric_args["arg4"];
      alloc->argv.mixed_pool.dynamic_initial_alloc_bytes = m_event.numeric_args["arg5"];
      alloc->argv.mixed_pool.dynamic_min_alloc_bytes = m_event.numeric_args["arg6"];
    }
    else if (m_event.numeric_args.find("arg5") != m_event.numeric_args.end()) {
      alloc->argc = 6;
      alloc->argv.mixed_pool.smallest_fixed_blocksize = m_event.numeric_args["arg1"];
      alloc->argv.mixed_pool.largest_fixed_blocksize = m_event.numeric_args["arg2"];
      alloc->argv.mixed_pool.max_fixed_blocksize = m_event.numeric_args["arg3"];
      alloc->argv.mixed_pool.size_multiplier = m_event.numeric_args["arg4"];
      alloc->argv.mixed_pool.dynamic_initial_alloc_bytes = m_event.numeric_args["arg5"];
    }
    else if (m_event.numeric_args.find("arg4") != m_event.numeric_args.end()) {
      alloc->argc = 5;
      alloc->argv.mixed_pool.smallest_fixed_blocksize = m_event.numeric_args["arg1"];
      alloc->argv.mixed_pool.largest_fixed_blocksize = m_event.numeric_args["arg2"];
      alloc->argv.mixed_pool.max_fixed_blocksize = m_event.numeric_args["arg3"];
      alloc->argv.mixed_pool.size_multiplier = m_event.numeric_args["arg4"];
    }
    else if (m_event.numeric_args.find("arg3") != m_event.numeric_args.end()) {
      alloc->argc = 4;
      alloc->argv.mixed_pool.smallest_fixed_blocksize = m_event.numeric_args["arg1"];
      alloc->argv.mixed_pool.largest_fixed_blocksize = m_event.numeric_args["arg2"];
      alloc->argv.mixed_pool.max_fixed_blocksize = m_event.numeric_args["arg3"];
    }
    else if (m_event.numeric_args.find("arg2") != m_event.numeric_args.end()) {
      alloc->argc = 3;
      alloc->argv.mixed_pool.smallest_fixed_blocksize = m_event.numeric_args["arg1"];
      alloc->argv.mixed_pool.largest_fixed_blocksize = m_event.numeric_args["arg2"];
    }
    else if (m_event.numeric_args.find("arg1") != m_event.numeric_args.end()) {
      alloc->argc = 2;
      alloc->argv.mixed_pool.smallest_fixed_blocksize = m_event.numeric_args["arg1"];
    }
  }
  else {
    REPLAY_ERROR("Unknown class (" << type << "), skipping.");
  }

  const std::string allocator_ref_string{m_event.string_args["allocator_ref"]};
  const uint64_t obj_p{ getPointer(allocator_ref_string) };

  m_allocator_indices[obj_p] = hdr->num_allocators;

  ReplayFile::Operation* op = &hdr->ops[hdr->num_operations];
  memset(op, 0, sizeof(*op));

  op->op_type = ReplayFile::otype::ALLOCATOR_CREATION;
  op->op_line_number = m_line_number;
  op->op_allocator = hdr->num_allocators;

  m_allocator_index[allocator_name] = hdr->num_allocators;

  hdr->num_allocators++;
  if (hdr->num_allocators >= ReplayFile::max_allocators) {
    REPLAY_ERROR("Too many allocators for replay: " << hdr->num_allocators);
  }
  hdr->num_operations++;
}

void ReplayInterpreter::compile_set_default_allocator()
{
  const std::string allocator_ref_string{m_event.string_args["allocator_ref"]};
  ReplayFile::Header* hdr = m_ops->getOperationsTable();
  ReplayFile::Operation* op = &hdr->ops[hdr->num_operations];

  memset(op, 0, sizeof(*op));
  op->op_type = ReplayFile::otype::SETDEFAULTALLOCATOR;
  op->op_line_number = m_line_number;
  op->op_allocator = getAllocatorIndex(allocator_ref_string);
  hdr->num_operations++;
}

int ReplayInterpreter::getAllocatorIndex(const std::string& ref_s)
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

void ReplayInterpreter::compile_allocate()
{
  if (m_replaying_reallocate)
    return;

  ReplayFile::Header* hdr{m_ops->getOperationsTable()};
  ReplayFile::Operation* op{&hdr->ops[hdr->num_operations]};
  const std::string allocator_ref{m_event.string_args["allocator_ref"]};
  const std::size_t allocation_size{m_event.numeric_args["size"]};
  std::string pointer_string{m_event.string_args["pointer"]};
  const std::string pointer_key{allocator_ref + pointer_string};

  memset(op, 0, sizeof(*op));

  op->op_type = ReplayFile::otype::ALLOCATE;
  op->op_line_number = m_line_number;
  op->op_allocator = getAllocatorIndex(allocator_ref);
  op->op_size = allocation_size;

  if ( m_allocation_id.find(pointer_key) != m_allocation_id.end() ) {
    REPLAY_ERROR("Pointer already allocated: " << m_ops->getLine(m_line_number) << std::endl);
  }

  op->op_line_number = m_line_number;
  m_allocation_id.insert({pointer_key, hdr->num_operations});
  hdr->num_operations++;
}

void ReplayInterpreter::compile_reallocate()
{
  const std::string      allocator_ref{m_event.string_args["allocator_ref"]};
  ReplayFile::Header*    hdr{m_ops->getOperationsTable()};
  ReplayFile::Operation* op{&hdr->ops[hdr->num_operations]};

  if (m_event.string_args.find("new_ptr") == m_event.string_args.end() ) {
    //
    // First of two reallocate replays
    //
    const std::size_t allocation_size{m_event.numeric_args["size"]};
    const std::string current_ptr_string{m_event.string_args["current_ptr"]};
    const std::string current_ptr_key{allocator_ref + current_ptr_string};
    const uint64_t current_ptr{ getPointer(current_ptr_string) };

    m_replaying_reallocate = true;

    if ( current_ptr != 0 && (m_allocation_id.find(current_ptr_key) == m_allocation_id.end()) ) {
        REPLAY_ERROR("Rogue: " << m_ops->getLine(m_line_number) << std::endl);
    }

    memset(op, 0, sizeof(*op));
    op->op_type = ReplayFile::otype::REALLOCATE_EX;
    op->op_line_number = m_line_number;
    op->op_alloc_ops[1] = (current_ptr == 0) ? 0 : m_allocation_id[current_ptr_key];
    op->op_size = allocation_size;
    op->op_allocator = getAllocatorIndex(allocator_ref);

    if (current_ptr != 0)
      m_allocation_id.erase(current_ptr_key);
  }
  else {
    const std::string new_ptr_string{m_event.string_args["new_ptr"]};
    const std::string new_ptr_key{allocator_ref + new_ptr_string};

    if ( m_allocation_id.find(new_ptr_key) != m_allocation_id.end() ) {
      REPLAY_ERROR("Pointer already allocated: " << m_ops->getLine(m_line_number) << std::endl);
    }
    m_allocation_id.insert({new_ptr_key, hdr->num_operations});
    hdr->num_operations++;
    m_replaying_reallocate = false;
  }
}

bool ReplayInterpreter::compile_deallocate()
{
  const std::string allocator_ref{m_event.string_args["allocator_ref"]};
  const std::string memory_ptr_string{m_event.string_args["pointer"]};
  const std::string memory_ptr_key{allocator_ref + memory_ptr_string};

  if (m_replaying_reallocate)
    return false;

  ReplayFile::Header* hdr = m_ops->getOperationsTable();

  if ( m_allocation_id.find(memory_ptr_key) == m_allocation_id.end() ) {
    int id{getAllocatorIndex(allocator_ref)};

    std::cerr
      << "[IGNORED] Rogue deallocate ptr= " << memory_ptr_string
      << ", base_name=" << hdr->allocators[id].base_name
      << ", name= " << hdr->allocators[id].name << " "
      << m_ops->getLine(m_line_number)
      << std::endl;

    m_deallocate_rogue_ignored++;
    return false;
  }

  ReplayFile::Operation* op = &hdr->ops[hdr->num_operations];

  memset(op, 0, sizeof(*op));

  op->op_type = ReplayFile::otype::DEALLOCATE;
  op->op_line_number = m_line_number;
  op->op_allocator = getAllocatorIndex(allocator_ref);
  op->op_alloc_ops[0] = m_allocation_id[memory_ptr_key];
  hdr->num_operations++;

  m_allocation_id.erase(memory_ptr_key);
  return true;
}

void ReplayInterpreter::compile_coalesce()
{
  std::string allocator_name{m_event.tags["allocator_name"]};
  strip_off_base(allocator_name);

  ReplayFile::Header* hdr = m_ops->getOperationsTable();
  ReplayFile::Operation* op = &hdr->ops[hdr->num_operations];

  memset(op, 0, sizeof(*op));
  op->op_type = ReplayFile::otype::COALESCE;
  op->op_line_number = m_line_number;
  op->op_allocator = m_allocator_index[allocator_name];
  hdr->num_operations++;
}

void ReplayInterpreter::compile_release()
{
  const std::string allocator_ref_string{m_event.string_args["allocator_ref"]};
  ReplayFile::Header* hdr = m_ops->getOperationsTable();
  ReplayFile::Operation* op = &hdr->ops[hdr->num_operations];

  memset(op, 0, sizeof(*op));
  op->op_type = ReplayFile::otype::RELEASE;
  op->op_line_number = m_line_number;
  op->op_allocator = getAllocatorIndex(allocator_ref_string);
  hdr->num_operations++;
}
#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)

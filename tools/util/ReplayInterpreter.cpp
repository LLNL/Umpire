//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <fstream>
#include <sstream>
#include <string>

#include "util/ReplayInterpreter.hpp"
#include "util/ReplayMacros.hpp"
#include "util/ReplayOperationManager.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/tpl/json/json.hpp"
#include "umpire/util/AllocationRecord.hpp"

void ReplayInterpreter::runOperations(bool gather_statistics)
{
  m_operation_mgr.runOperations(gather_statistics);
}

void ReplayInterpreter::buildAllocMapOperations(void)
{
  while ( std::getline(m_input_file, m_line) ) {
    const std::string header("{ \"kind\":\"replay\", \"uid\":");
    auto header_len = header.size();

    if ( m_line.size() <= header_len || m_line.substr(0, header_len) != header.substr(0, header_len) )
      continue;

    m_json.clear();
    m_json = nlohmann::json::parse(m_line);

    if (   m_json["event"] == "makeAllocator"
        || m_json["event"] == "makeMemoryResource"
        || m_json["event"] == "allocate"
        || m_json["event"] == "deallocate"
        || m_json["event"] == "coalesce"
        || m_json["event"] == "release"
        || m_json["event"] == "version"
    ) {
      continue;
    }

    ++m_op_seq;
    compare_ss.str("");
    compare_ss << m_json["event"] << " ";

    if ( m_json["event"] == "allocation_map_insert" ) {
      replay_makeAllocationMapInsert();
    }
    else if ( m_json["event"] == "allocation_map_find" ) {
      replay_makeAllocationMapFind();
    }
    else if ( m_json["event"] == "allocation_map_remove" ) {
      replay_makeAllocationMapRemove();
    }
    else if ( m_json["event"] == "allocation_map_clear" ) {
      replay_makeAllocationMapClear();
    }
    else {
      REPLAY_ERROR("Unknown Replay (" << m_json["event"] << ")");
    }
    compare_ss << std::endl;
  }
}

void ReplayInterpreter::buildOperations(void)
{
  while ( std::getline(m_input_file, m_line) ) {
    const std::string header("{ \"kind\":\"replay\", \"uid\":");
    auto header_len = header.size();

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

    ++m_op_seq;
    compare_ss.str("");
    compare_ss << m_json["event"] << " ";

    if ( m_json["event"] == "makeAllocator" ) {
      replay_makeAllocator();
    }
    else if ( m_json["event"] == "makeMemoryResource" ) {
      replay_makeMemoryResource();
    }
    else if ( m_json["event"] == "allocate" ) {
      replay_allocate();
    }
    else if ( m_json["event"] == "deallocate" ) {
      replay_deallocate();
    }
    else if ( m_json["event"] == "coalesce" ) {
      replay_coalesce();
    }
    else if ( m_json["event"] == "release" ) {
      replay_release();
    }
    else if ( m_json["event"] == "version" ) {
      if (   m_json["payload"]["major"] != UMPIRE_VERSION_MAJOR
          || m_json["payload"]["minor"] != UMPIRE_VERSION_MINOR
          || m_json["payload"]["patch"] != UMPIRE_VERSION_PATCH ) {

        REPLAY_WARNING("Warning, version mismatch:\n"
          << "  Tool version: " << UMPIRE_VERSION_MAJOR << "." << UMPIRE_VERSION_MINOR << "." << UMPIRE_VERSION_PATCH << std::endl
          << "  Log  version: "
          << m_json["payload"]["major"] << "."
          << m_json["payload"]["minor"]  << "."
          << m_json["payload"]["patch"]);

        if (m_json["payload"]["major"] != UMPIRE_VERSION_MAJOR) {
          REPLAY_ERROR("Warning, major version mismatch:\n"
            << "  Tool version: " << UMPIRE_VERSION_MAJOR << "." << UMPIRE_VERSION_MINOR << "." << UMPIRE_VERSION_PATCH << std::endl
            << "  Log  version: "
            << m_json["payload"]["major"] << "."
            << m_json["payload"]["minor"]  << "."
            << m_json["payload"]["patch"]);
        }
      }
    }
    else {
      REPLAY_ERROR("Unknown Replay (" << m_json["event"] << ")");
    }
    compare_ss << std::endl;
  }
}

//
// Return: > 0 success, 0 eof, < 0 error
//
int ReplayInterpreter::getSymbolicOperation( std::string& raw_line, std::string& sym_line )
{
  while ( std::getline(m_input_file, m_line) ) {
    const std::string header("{ \"kind\":\"replay\", \"uid\":");
    auto header_len = header.size();

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

    ++m_op_seq;
    compare_ss.str("");
    compare_ss << m_json["event"] << " ";

    if ( m_json["event"] == "makeAllocator" ) {
      replay_makeAllocator();
    }
    else if ( m_json["event"] == "makeMemoryResource" ) {
      replay_makeMemoryResource();
    }
    else if ( m_json["event"] == "allocate" ) {
      replay_allocate();
    }
    else if ( m_json["event"] == "deallocate" ) {
      replay_deallocate();
    }
    else if ( m_json["event"] == "coalesce" ) {
      replay_coalesce();
    }
    else if ( m_json["event"] == "release" ) {
      replay_release();
    }
    else if ( m_json["event"] == "version" ) {
      if (   m_json["payload"]["major"] != UMPIRE_VERSION_MAJOR
          || m_json["payload"]["minor"] != UMPIRE_VERSION_MINOR
          || m_json["payload"]["patch"] != UMPIRE_VERSION_PATCH ) {

        REPLAY_WARNING("Warning, version mismatch:\n"
          << "  Tool version: " << UMPIRE_VERSION_MAJOR << "." << UMPIRE_VERSION_MINOR << "." << UMPIRE_VERSION_PATCH << std::endl
          << "  Log  version: "
          << m_json["payload"]["major"] << "."
          << m_json["payload"]["minor"]  << "."
          << m_json["payload"]["patch"]);

        if (m_json["payload"]["major"] != UMPIRE_VERSION_MAJOR) {
          REPLAY_ERROR("Warning, major version mismatch:\n"
            << "  Tool version: " << UMPIRE_VERSION_MAJOR << "." << UMPIRE_VERSION_MINOR << "." << UMPIRE_VERSION_PATCH << std::endl
            << "  Log  version: "
            << m_json["payload"]["major"] << "."
            << m_json["payload"]["minor"]  << "."
            << m_json["payload"]["patch"]);
        }
      }
    }
    else {
      REPLAY_ERROR("Unknown Replay (" << m_json["event"] << ")");
    }

    compare_ss << std::endl;
    raw_line = m_line;
    sym_line = compare_ss.str();
    return 1;
  }
  return 0;   // EOF
}

ReplayInterpreter::ReplayInterpreter( std::string in_file_name ):
    m_input_file(in_file_name), m_num_allocators(0), m_op_seq(0)
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

void ReplayInterpreter::replay_makeMemoryResource( void )
{
  const std::string& allocator_name = m_json["payload"]["name"];
  const std::string& obj_s = m_json["result"];
  const uint64_t obj_p = std::stoul(obj_s, nullptr, 0);

  m_allocator_indices[obj_p] = m_num_allocators++;
  compare_ss  << allocator_name
              << " " << m_allocator_indices[obj_p];

  m_operation_mgr.makeMemoryResource(allocator_name);
}

void ReplayInterpreter::replay_makeAllocator( void )
{
  const std::string& allocator_name = m_json["payload"]["allocator_name"];

  //
  // When the result isn't set, just perform the operation.  We will
  // establish the mapping on after the result has been recorded in the
  // two-step REPLAY process for this event.
  //
  if ( m_json["result"].is_null() ) {
    const bool introspection = m_json["payload"]["with_introspection"];
    const std::string type = m_json["payload"]["type"];

    if ( type == "umpire::strategy::AllocationAdvisor" ) {
      const int numargs = static_cast<int>(m_json["payload"]["args"].size());
      const std::string& base_allocator_name = m_json["payload"]["args"][0];
      const std::string& advice_operation = m_json["payload"]["args"][1];
      const std::string& last_arg = m_json["payload"]["args"][numargs-1];

      //
      // The last argument to this constructor will either be a string or
      // will be an integer.  If it is an integer, we assume that it is the
      // optional device_id argument.
      //
      int device_id = -1;   // Use default argument if negative
      if (last_arg.find_first_not_of( "0123456789" ) == std::string::npos) {
        std::stringstream ss(last_arg);
        ss >> device_id;
      }

      if (device_id >= 0) { // Optional device ID specified
        switch ( numargs ) {
        default:
          REPLAY_ERROR("Invalid number of arguments (" << numargs
            << " for " << type << " operation.  Stopping");
        case 3:
          compare_ss << introspection 
            << " " << allocator_name 
            << " " << base_allocator_name
            << " " << advice_operation 
            << " " << device_id
          ;
          m_operation_mgr.makeAdvisor(
              introspection, allocator_name, base_allocator_name,
              advice_operation, device_id);
          break;
        case 4:
          const std::string& accessing_allocator_name = m_json["payload"]["args"][2];

          compare_ss << introspection 
            << " " << allocator_name 
            << " " << base_allocator_name
            << " " << advice_operation 
            << " " << accessing_allocator_name 
            << " " << device_id
          ;
          m_operation_mgr.makeAdvisor(
              introspection, allocator_name, base_allocator_name,
              advice_operation, accessing_allocator_name, device_id);
          break;
        }
      }
      else { // Use default device_id
        switch ( numargs ) {
        default:
          REPLAY_ERROR("Invalid number of arguments (" << numargs
            << " for " << type << " operation.  Stopping");
        case 2:
          compare_ss << introspection 
            << " " << allocator_name 
            << " " << base_allocator_name
            << " " << advice_operation 
          ;
          m_operation_mgr.makeAdvisor(
              introspection, allocator_name, base_allocator_name,
              advice_operation);
          break;
        case 3:
          const std::string& accessing_allocator_name = m_json["payload"]["args"][2];

          compare_ss << introspection 
            << " " << allocator_name 
            << " " << base_allocator_name
            << " " << advice_operation 
            << " " << accessing_allocator_name 
          ;
          m_operation_mgr.makeAdvisor(
              introspection, allocator_name, base_allocator_name,
              advice_operation, accessing_allocator_name);
          break;
        }
      }
    }
    else if ( type == "umpire::strategy::DynamicPoolList" ) {
      const std::string& base_allocator_name = m_json["payload"]["args"][0];

      std::size_t initial_alloc_size;
      std::size_t min_alloc_size;

      // Now grab the optional fields
      if (m_json["payload"]["args"].size() >= 3) {
        get_from_string(m_json["payload"]["args"][1], initial_alloc_size);
        get_from_string(m_json["payload"]["args"][2], min_alloc_size);

        compare_ss << introspection 
          << " " << allocator_name 
          << " " << base_allocator_name
          << " " << initial_alloc_size 
          << " " << min_alloc_size 
        ;
        m_operation_mgr.makeDynamicPoolList(
              introspection
            , allocator_name
            , base_allocator_name
            , initial_alloc_size
            , min_alloc_size
            , umpire::strategy::heuristic_percent_releasable_list(0)
        );
      }
      else if (m_json["payload"]["args"].size() == 2) {
        get_from_string(m_json["payload"]["args"][1], initial_alloc_size);

        compare_ss << introspection 
          << " " << allocator_name 
          << " " << base_allocator_name
          << " " << initial_alloc_size 
        ;
        m_operation_mgr.makeDynamicPoolList(
              introspection
            , allocator_name
            , base_allocator_name
            , initial_alloc_size
        );
      }
      else {
        compare_ss << introspection 
          << " " << allocator_name 
          << " " << base_allocator_name
        ;
        m_operation_mgr.makeDynamicPoolList(
              introspection
            , allocator_name
            , base_allocator_name
        );
      }
    }
    else if ( type == "umpire::strategy::DynamicPool" || type == "umpire::strategy::DynamicPoolMap" ) {
      const std::string& base_allocator_name = m_json["payload"]["args"][0];

      std::size_t initial_alloc_size;
      std::size_t min_alloc_size;
      int alignment;

      // Now grab the optional fields
      if (m_json["payload"]["args"].size() >= 4) {
        get_from_string(m_json["payload"]["args"][1], initial_alloc_size);
        get_from_string(m_json["payload"]["args"][2], min_alloc_size);
        get_from_string(m_json["payload"]["args"][3], alignment);

        compare_ss << introspection 
          << " " << allocator_name 
          << " " << base_allocator_name
          << " " << initial_alloc_size 
          << " " << min_alloc_size 
          << " " << alignment 
        ;
        m_operation_mgr.makeDynamicPoolMap(
              introspection
            , allocator_name
            , base_allocator_name
            , initial_alloc_size
            , min_alloc_size
            , umpire::strategy::heuristic_percent_releasable(0)
            , alignment
        );
      }
      else if (m_json["payload"]["args"].size() >= 3) {
        get_from_string(m_json["payload"]["args"][1], initial_alloc_size);
        get_from_string(m_json["payload"]["args"][2], min_alloc_size);

        compare_ss << introspection 
          << " " << allocator_name 
          << " " << base_allocator_name
          << " " << initial_alloc_size 
          << " " << min_alloc_size 
        ;
        m_operation_mgr.makeDynamicPoolMap(
              introspection
            , allocator_name
            , base_allocator_name
            , initial_alloc_size
            , min_alloc_size
            , umpire::strategy::heuristic_percent_releasable(0)
        );
      }
      else if (m_json["payload"]["args"].size() == 2) {
        get_from_string(m_json["payload"]["args"][1], initial_alloc_size);

        compare_ss << introspection 
          << " " << allocator_name 
          << " " << base_allocator_name
          << " " << initial_alloc_size 
        ;
        m_operation_mgr.makeDynamicPoolMap(
              introspection
            , allocator_name
            , base_allocator_name
            , initial_alloc_size
        );
      }
      else {
        compare_ss << introspection 
          << " " << allocator_name 
          << " " << base_allocator_name
        ;
        m_operation_mgr.makeDynamicPoolMap(
              introspection
            , allocator_name
            , base_allocator_name
        );
      }
    }
    else if ( type == "umpire::strategy::MonotonicAllocationStrategy" ) {
      const std::string& base_allocator_name = m_json["payload"]["args"][1];

      std::size_t capacity;
      get_from_string(m_json["payload"]["args"][0], capacity);

      compare_ss << introspection 
        << " " << allocator_name 
        << " " << capacity
        << " " << base_allocator_name
      ;
      m_operation_mgr.makeMonotonicAllocator(
            introspection
          , allocator_name
          , capacity
          , base_allocator_name
      );
    }
    else if ( type == "umpire::strategy::SlotPool" ) {
      const std::string& base_allocator_name = m_json["payload"]["args"][1];

      std::size_t slots;
      get_from_string(m_json["payload"]["args"][0], slots);

      compare_ss << introspection 
        << " " << allocator_name 
        << " " << slots
        << " " << base_allocator_name
      ;
      m_operation_mgr.makeSlotPool(
            introspection
          , allocator_name
          , slots
          , base_allocator_name
      );
    }
    else if ( type == "umpire::strategy::SizeLimiter" ) {
      const std::string& base_allocator_name = m_json["payload"]["args"][0];
      std::size_t size_limit;
      get_from_string(m_json["payload"]["args"][1], size_limit);

      compare_ss << introspection 
        << " " << allocator_name 
        << " " << base_allocator_name
        << " " << size_limit
      ;
      m_operation_mgr.makeSizeLimiter(
            introspection
          , allocator_name
          , base_allocator_name
          , size_limit
      );
    }
    else if ( type == "umpire::strategy::ThreadSafeAllocator" ) {
      const std::string& base_allocator_name = m_json["payload"]["args"][0];

      compare_ss << introspection 
        << " " << allocator_name 
        << " " << base_allocator_name
      ;
      m_operation_mgr.makeThreadSafeAllocator(
            introspection
          , allocator_name
          , base_allocator_name
      );
    }
    else if ( type == "umpire::strategy::FixedPool" ) {
      const std::string& base_allocator_name = m_json["payload"]["args"][0];

      std::size_t object_bytes;
      std::size_t objects_per_pool;

      get_from_string(m_json["payload"]["args"][1], object_bytes);

      // Now grab the optional fields
      if (m_json["payload"]["args"].size() >= 3) {
        get_from_string(m_json["payload"]["args"][2], objects_per_pool);

        compare_ss << introspection 
          << " " << allocator_name 
          << " " << base_allocator_name
          << " " << object_bytes 
          << " " << objects_per_pool 
        ;
        m_operation_mgr.makeFixedPool(
              introspection
            , allocator_name
            , base_allocator_name
            , object_bytes
            , objects_per_pool
        );
      }
      else {
        compare_ss << introspection 
          << " " << allocator_name 
          << " " << base_allocator_name
          << " " << object_bytes 
        ;
        m_operation_mgr.makeFixedPool(
              introspection
            , allocator_name
            , base_allocator_name
            , object_bytes
        );
      }
    }
    else if ( type == "umpire::strategy::MixedPool" ) {
      const std::string& base_allocator_name = m_json["payload"]["args"][0];
      std::size_t smallest_fixed_blocksize;
      std::size_t largest_fixed_blocksize;
      std::size_t max_fixed_blocksize;
      std::size_t size_multiplier;
      std::size_t dynamic_initial_alloc_bytes;
      std::size_t dynamic_min_alloc_bytes;
      int dynamic_align_bytes;

      // Now grab the optional fields
      if (m_json["payload"]["args"].size() >= 8) {
        get_from_string(m_json["payload"]["args"][1], smallest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][2], largest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][3], max_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][4], size_multiplier);
        get_from_string(m_json["payload"]["args"][5], dynamic_initial_alloc_bytes);
        get_from_string(m_json["payload"]["args"][6], dynamic_min_alloc_bytes);
        get_from_string(m_json["payload"]["args"][7], dynamic_align_bytes);

        compare_ss << introspection
            << " " << allocator_name
            << " " << base_allocator_name
            << " " << smallest_fixed_blocksize
            << " " << largest_fixed_blocksize
            << " " << max_fixed_blocksize
            << " " << size_multiplier
            << " " << dynamic_initial_alloc_bytes
            << " " << dynamic_min_alloc_bytes
            << " " << dynamic_align_bytes
        ;

        m_operation_mgr.makeMixedPool(
            introspection, allocator_name, base_allocator_name
          , smallest_fixed_blocksize
          , largest_fixed_blocksize
          , max_fixed_blocksize
          , size_multiplier
          , dynamic_initial_alloc_bytes
          , dynamic_min_alloc_bytes
          , umpire::strategy::heuristic_percent_releasable(0)
          , dynamic_align_bytes
        );
      }
      else if (m_json["payload"]["args"].size() >= 7) {
        get_from_string(m_json["payload"]["args"][1], smallest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][2], largest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][3], max_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][4], size_multiplier);
        get_from_string(m_json["payload"]["args"][5], dynamic_initial_alloc_bytes);
        get_from_string(m_json["payload"]["args"][6], dynamic_min_alloc_bytes);

        compare_ss << introspection
            << " " << allocator_name
            << " " << base_allocator_name
            << " " << smallest_fixed_blocksize
            << " " << largest_fixed_blocksize
            << " " << max_fixed_blocksize
            << " " << size_multiplier
            << " " << dynamic_initial_alloc_bytes
            << " " << dynamic_min_alloc_bytes
        ;

        m_operation_mgr.makeMixedPool(
            introspection, allocator_name, base_allocator_name
          , smallest_fixed_blocksize
          , largest_fixed_blocksize
          , max_fixed_blocksize
          , size_multiplier
          , dynamic_initial_alloc_bytes
          , dynamic_min_alloc_bytes
          , umpire::strategy::heuristic_percent_releasable(0)
        );
      }
      else if (m_json["payload"]["args"].size() >= 6) {
        get_from_string(m_json["payload"]["args"][1], smallest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][2], largest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][3], max_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][4], size_multiplier);
        get_from_string(m_json["payload"]["args"][5], dynamic_initial_alloc_bytes);

        compare_ss << introspection
            << " " << allocator_name
            << " " << base_allocator_name
            << " " << smallest_fixed_blocksize
            << " " << largest_fixed_blocksize
            << " " << max_fixed_blocksize
            << " " << size_multiplier
            << " " << dynamic_initial_alloc_bytes
        ;
        m_operation_mgr.makeMixedPool(
            introspection, allocator_name, base_allocator_name
          , smallest_fixed_blocksize
          , largest_fixed_blocksize
          , max_fixed_blocksize
          , size_multiplier
          , dynamic_initial_alloc_bytes
        );
      }
      else if (m_json["payload"]["args"].size() >= 5) {
        get_from_string(m_json["payload"]["args"][1], smallest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][2], largest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][3], max_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][4], size_multiplier);

        compare_ss << introspection
            << " " << allocator_name
            << " " << base_allocator_name
            << " " << smallest_fixed_blocksize
            << " " << largest_fixed_blocksize
            << " " << max_fixed_blocksize
            << " " << size_multiplier
        ;
        m_operation_mgr.makeMixedPool(
            introspection, allocator_name, base_allocator_name
          , smallest_fixed_blocksize
          , largest_fixed_blocksize
          , max_fixed_blocksize
          , size_multiplier
        );
      }
      else if (m_json["payload"]["args"].size() >= 4) {
        get_from_string(m_json["payload"]["args"][1], smallest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][2], largest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][3], max_fixed_blocksize);

        compare_ss << introspection
            << " " << allocator_name
            << " " << base_allocator_name
            << " " << smallest_fixed_blocksize
            << " " << largest_fixed_blocksize
            << " " << max_fixed_blocksize
        ;
        m_operation_mgr.makeMixedPool(
            introspection, allocator_name, base_allocator_name
          , smallest_fixed_blocksize
          , largest_fixed_blocksize
          , max_fixed_blocksize
        );
      }
      else if (m_json["payload"]["args"].size() >= 3) {
        get_from_string(m_json["payload"]["args"][1], smallest_fixed_blocksize);
        get_from_string(m_json["payload"]["args"][2], largest_fixed_blocksize);

        compare_ss << introspection
            << " " << allocator_name
            << " " << base_allocator_name
            << " " << smallest_fixed_blocksize
            << " " << largest_fixed_blocksize
        ;
        m_operation_mgr.makeMixedPool(
            introspection, allocator_name, base_allocator_name
          , smallest_fixed_blocksize
          , largest_fixed_blocksize
        );
      }
      else if (m_json["payload"]["args"].size() >= 2) {
        get_from_string(m_json["payload"]["args"][1], smallest_fixed_blocksize);

        compare_ss << introspection
            << " " << allocator_name
            << " " << base_allocator_name
            << " " << smallest_fixed_blocksize
        ;
        m_operation_mgr.makeMixedPool(
            introspection, allocator_name, base_allocator_name
          , smallest_fixed_blocksize
        );
      }
      else {
        compare_ss << introspection
            << " " << allocator_name
            << " " << base_allocator_name
        ;
        m_operation_mgr.makeMixedPool(
            introspection, allocator_name, base_allocator_name
        );
      }
    }
    else {
      REPLAY_ERROR("Unknown class (" << type << "), skipping.");
    }
  }
  else {
    const std::string obj_s = m_json["result"]["allocator_ref"];
    const uint64_t obj_p = std::stoul(obj_s, nullptr, 0);

    m_allocator_indices[obj_p] = m_num_allocators++;
    compare_ss << m_allocator_indices[obj_p];

    m_operation_mgr.makeAllocatorCont();
  }
}

void ReplayInterpreter::replay_allocate( void )
{
  const std::string alloc_obj_s = m_json["payload"]["allocator_ref"];
  const uint64_t alloc_obj_p = std::stoul(alloc_obj_s, nullptr, 0);
  auto n_iter = m_allocator_indices.find(alloc_obj_p);

  if ( n_iter == m_allocator_indices.end() )
    REPLAY_ERROR("Unknown allocator " << (void*)alloc_obj_p);

  const AllocatorIndex& allocator_number = n_iter->second;

  //
  // For allocations, two records are written.  The first record simply
  // contents the intention of the allocation (e.g. allocator and size).
  // The second record contains the result of the operation.
  //
  // We separate the intention from the result so that we may attempt to
  // replay an operation that may have failed
  //
  if ( m_json["result"].is_null() ) {
    const std::size_t alloc_size = m_json["payload"]["size"];

    compare_ss << allocator_number << " " << alloc_size;
    m_operation_mgr.makeAllocate(allocator_number, alloc_size);
  }
  else {
    const std::string memory_str = m_json["result"]["memory_ptr"];
    const uint64_t memory_ptr = std::stoul(memory_str, nullptr, 0);

    compare_ss << m_op_seq;
    m_allocation_id[memory_ptr] = m_op_seq;
    m_operation_mgr.makeAllocateCont(memory_ptr);
  }
}

void ReplayInterpreter::replay_deallocate( void )
{
  const std::string alloc_obj_s = m_json["payload"]["allocator_ref"];
  const uint64_t alloc_obj_p = std::stoul(alloc_obj_s, nullptr, 0);
  auto n_iter = m_allocator_indices.find(alloc_obj_p);

  if ( n_iter == m_allocator_indices.end() )
    REPLAY_ERROR("Unable to find allocator for: " << m_json["payload"]["memory_ptr"] << " deallocation ignored");

  const AllocatorIndex& allocator_number = n_iter->second;

  const std::string memory_str = m_json["payload"]["memory_ptr"];
  const uint64_t memory_ptr = std::stoul(memory_str, nullptr, 0);

  compare_ss << allocator_number << " " << m_allocation_id[memory_ptr];

  m_operation_mgr.makeDeallocate(allocator_number, memory_ptr);
}

void ReplayInterpreter::replay_coalesce( void )
{
  std::string allocator_name = m_json["payload"]["allocator_name"];
  strip_off_base(allocator_name);

  compare_ss << allocator_name;
  m_operation_mgr.makeCoalesce(allocator_name);
}

void ReplayInterpreter::replay_release( void )
{
  const std::string alloc_obj_s = m_json["payload"]["allocator_ref"];
  const uint64_t alloc_obj_p = std::stoul(alloc_obj_s, nullptr, 0);
  auto n_iter = m_allocator_indices.find(alloc_obj_p);

  if ( n_iter == m_allocator_indices.end() )
    REPLAY_ERROR("Unable to find allocator for: " << m_json["payload"]["memory_ptr"] << " release ignored");

  const AllocatorIndex& allocator_number = n_iter->second;
  compare_ss << allocator_number;
  m_operation_mgr.makeRelease(allocator_number);
}

void ReplayInterpreter::replay_makeAllocationMapInsert( void )
{
  const std::string key_s = m_json["payload"]["ptr"];
  void* key = reinterpret_cast<void*>(std::stoul(key_s, nullptr, 0));
  const std::string rec_ptr_s = m_json["payload"]["record_ptr"];
  const std::string rec_size_s = m_json["payload"]["record_size"];
  const std::string rec_strategy_s = m_json["payload"]["record_strategy"];

  umpire::util::AllocationRecord arec;
  arec.ptr = reinterpret_cast<void*>(std::stoul(rec_ptr_s, nullptr, 0));
  arec.size = reinterpret_cast<std::size_t>(std::stoul(rec_size_s, nullptr, 0));
  arec.strategy = reinterpret_cast<umpire::strategy::AllocationStrategy*>(std::stoul(rec_strategy_s, nullptr, 0));

  m_operation_mgr.makeAllocationMapInsert(key, arec);
}

void ReplayInterpreter::replay_makeAllocationMapFind( void )
{
  const std::string key_s = m_json["payload"]["ptr"];
  void* key = reinterpret_cast<void*>(std::stoul(key_s, nullptr, 0));

  m_operation_mgr.makeAllocationMapFind(key);
}

void ReplayInterpreter::replay_makeAllocationMapRemove( void )
{
  const std::string key_s = m_json["payload"]["ptr"];
  void* key = reinterpret_cast<void*>(std::stoul(key_s, nullptr, 0));

  m_operation_mgr.makeAllocationMapRemove(key);
}

void ReplayInterpreter::replay_makeAllocationMapClear( void )
{
  m_operation_mgr.makeAllocationMapClear();
}


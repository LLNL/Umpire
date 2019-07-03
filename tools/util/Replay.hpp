//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef REPLAY_Replay_HPP
#define REPLAY_Replay_HPP
#include "util/OperationManager.hpp"
#include "umpire/tpl/json/json.hpp"

class Replay {
  public:
    void run(void)
    {
      m_operation_mgr.run();
    }

    void build(void)
    {
      while ( std::getline(m_input_file, m_line) ) {
        const std::string header("{ \"kind\":\"replay\", \"uid\":");
        auto header_len = header.size();

        if ( m_line.size() <= header_len || m_line.substr(0, header_len) != header.substr(0, header_len) )
          continue;

        m_json.clear();
        m_json = nlohmann::json::parse(m_line);

        if ( m_json["event"] == "makeAllocator" ) {
          replay_makeAllocator();
        }
        else if ( m_json["event"] == "makeMemoryResource" ) {
          continue;
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
            std::cerr << "Warning, version mismatch:\n"
              << "  Tool version: " << UMPIRE_VERSION_MAJOR << "." << UMPIRE_VERSION_MINOR << "." << UMPIRE_VERSION_PATCH << std::endl
              << "  Log  version: "
              << m_json["payload"]["major"] << "."
              << m_json["payload"]["minor"]  << "."
              << m_json["payload"]["patch"]  << std::endl;
          }
        }
        else {
          std::cerr << "Unknown Replay (" << m_json["event"] << ")\n";
          exit (1);
        }
      }
    }

    Replay( std::string in_file_name ):
        m_input_file(in_file_name), m_num_allocators(0)
    {
      if ( ! m_input_file.is_open() ) {
        std::cerr << "Unable to open input file " << in_file_name << std::endl;
        exit (1);
      }
    }

  private:
    using AllocatorIndex = int;
    using AllocatorFromLog = uint64_t;
    using AllocatorIndexMap = std::unordered_map<AllocatorFromLog, AllocatorIndex>;

    std::ifstream m_input_file;
    std::unordered_map<std::string, void*> m_allocated_ptrs;    // key(alloc_ptr), val(replay_alloc_ptr)
    std::string m_line;
    nlohmann::json m_json;
    std::vector<std::string> m_row;
    AllocatorIndex m_num_allocators;
    AllocatorIndexMap m_allocator_indices;
    OperationManager m_operation_mgr;

    template <typename T> void
    get_from_string( const std::string& s, T& val )
    {
        std::istringstream ss(s);
        ss >> val;
    }

    void
    strip_off_base(std::string& s)
    {
      const std::string base("_base");

      if (s.length() > base.length()) {
        if (s.compare(s.length() - base.length(), base.length(), base) == 0) {
          s.erase(s.length() - base.length(), base.length());
        }
      }
    }

    void replay_makeAllocator( void )
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
            case 3:
              m_operation_mgr.bld_advisor(
                  introspection, allocator_name, base_allocator_name,
                  advice_operation, device_id);
              break;
            case 4:
              const std::string& accessing_allocator_name = m_json["payload"]["args"][2];

              m_operation_mgr.bld_advisor(
                  introspection, allocator_name, base_allocator_name,
                  advice_operation, accessing_allocator_name, device_id);
              break;
            }
          }
          else { // Use default device_id
            switch ( numargs ) {
            case 2:
              m_operation_mgr.bld_advisor(
                  introspection, allocator_name, base_allocator_name,
                  advice_operation);
              break;
            case 3:
              const std::string& accessing_allocator_name = m_json["payload"]["args"][2];

              m_operation_mgr.bld_advisor(
                  introspection, allocator_name, base_allocator_name,
                  advice_operation, accessing_allocator_name);
              break;
            }
          }
        }
        else if ( type == "umpire::strategy::DynamicPool" ) {
          const std::string& base_allocator_name = m_json["payload"]["args"][0];

          std::size_t min_initial_alloc_size;
          std::size_t min_alloc_size;

          // Now grab the optional fields
          if (m_json["payload"]["args"].size() >= 3) {
            get_from_string(m_json["payload"]["args"][1], min_initial_alloc_size);
            get_from_string(m_json["payload"]["args"][2], min_alloc_size);

            m_operation_mgr.bld_dynamicpool(
                  introspection
                , allocator_name
                , base_allocator_name
                , min_initial_alloc_size
                , min_alloc_size
                , umpire::strategy::heuristic_percent_releasable(0)
            );
          }
          else if (m_json["payload"]["args"].size() == 2) {
            get_from_string(m_json["payload"]["args"][1], min_initial_alloc_size);

            m_operation_mgr.bld_dynamicpool(
                  introspection
                , allocator_name
                , base_allocator_name
                , min_initial_alloc_size
            );
          }
          else {
            m_operation_mgr.bld_dynamicpool(
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

          m_operation_mgr.bld_monotonic(
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

          m_operation_mgr.bld_slotpool(
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

          m_operation_mgr.bld_limiter(
                introspection
              , allocator_name
              , base_allocator_name
              , size_limit
          );
        }
        else if ( type == "umpire::strategy::ThreadSafeAllocator" ) {
          const std::string& base_allocator_name = m_json["payload"]["args"][0];

          m_operation_mgr.bld_threadsafe(
                introspection
              , allocator_name
              , base_allocator_name
          );
        }
        else if ( type == "umpire::strategy::FixedPool" ) {
          //
          // Need to skip FixedPool for now since I haven't figured out how to
          // dynamically parse/creat the data type parameter
          //
          return;
        }
        else if ( type == "umpire::strategy::MixedPool" ) {
          const std::string& base_allocator_name = m_json["payload"]["args"][0];
          std::size_t smallest_fixed_blocksize;
          std::size_t largest_fixed_blocksize;
          std::size_t max_fixed_blocksize;
          std::size_t size_multiplier;
          std::size_t dynamic_min_initial_alloc_size;
          std::size_t dynamic_min_alloc_size;

          // Now grab the optional fields
          if (m_json["payload"]["args"].size() >= 7) {
            get_from_string(m_json["payload"]["args"][1], smallest_fixed_blocksize);
            get_from_string(m_json["payload"]["args"][2], largest_fixed_blocksize);
            get_from_string(m_json["payload"]["args"][3], max_fixed_blocksize);
            get_from_string(m_json["payload"]["args"][4], size_multiplier);
            get_from_string(m_json["payload"]["args"][5], dynamic_min_initial_alloc_size);
            get_from_string(m_json["payload"]["args"][6], dynamic_min_alloc_size);

            m_operation_mgr.bld_mixedpool(
                introspection, allocator_name, base_allocator_name
              , smallest_fixed_blocksize
              , largest_fixed_blocksize
              , max_fixed_blocksize
              , size_multiplier
              , dynamic_min_initial_alloc_size
              , dynamic_min_alloc_size
              , umpire::strategy::heuristic_percent_releasable(0)
            );
          }
          else if (m_json["payload"]["args"].size() >= 6) {
            get_from_string(m_json["payload"]["args"][1], smallest_fixed_blocksize);
            get_from_string(m_json["payload"]["args"][2], largest_fixed_blocksize);
            get_from_string(m_json["payload"]["args"][3], max_fixed_blocksize);
            get_from_string(m_json["payload"]["args"][4], size_multiplier);
            get_from_string(m_json["payload"]["args"][5], dynamic_min_initial_alloc_size);

            m_operation_mgr.bld_mixedpool(
                introspection, allocator_name, base_allocator_name
              , smallest_fixed_blocksize
              , largest_fixed_blocksize
              , max_fixed_blocksize
              , size_multiplier
              , dynamic_min_initial_alloc_size
            );
          }
          else if (m_json["payload"]["args"].size() >= 5) {
            get_from_string(m_json["payload"]["args"][1], smallest_fixed_blocksize);
            get_from_string(m_json["payload"]["args"][2], largest_fixed_blocksize);
            get_from_string(m_json["payload"]["args"][3], max_fixed_blocksize);
            get_from_string(m_json["payload"]["args"][4], size_multiplier);

            m_operation_mgr.bld_mixedpool(
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

            m_operation_mgr.bld_mixedpool(
                introspection, allocator_name, base_allocator_name
              , smallest_fixed_blocksize
              , largest_fixed_blocksize
              , max_fixed_blocksize
            );
          }
          else if (m_json["payload"]["args"].size() >= 3) {
            get_from_string(m_json["payload"]["args"][1], smallest_fixed_blocksize);
            get_from_string(m_json["payload"]["args"][2], largest_fixed_blocksize);

            m_operation_mgr.bld_mixedpool(
                introspection, allocator_name, base_allocator_name
              , smallest_fixed_blocksize
              , largest_fixed_blocksize
            );
          }
          else if (m_json["payload"]["args"].size() >= 2) {
            get_from_string(m_json["payload"]["args"][1], smallest_fixed_blocksize);

            m_operation_mgr.bld_mixedpool(
                introspection, allocator_name, base_allocator_name
              , smallest_fixed_blocksize
            );
          }
          else {
            m_operation_mgr.bld_mixedpool(
                introspection, allocator_name, base_allocator_name
            );
          }
        }
        else {
          std::cerr << "Unknown class (" << type << "), skipping.\n";
          return;
        }
      }
      else {
        const std::string obj_s = m_json["result"]["allocator_ref"];
        const uint64_t obj_p = std::stoul(obj_s, nullptr, 0);

        m_allocator_indices[obj_p] = m_num_allocators++;

        m_operation_mgr.bld_allocator_cont();
      }
    }

    void replay_allocate( void )
    {
      const std::string alloc_obj_s = m_json["payload"]["allocator_ref"];
      const uint64_t alloc_obj_p = std::stoul(alloc_obj_s, nullptr, 0);
      auto n_iter = m_allocator_indices.find(alloc_obj_p);

      if ( n_iter == m_allocator_indices.end() ) {
        std::cerr << "Unknown allocator " << (void*)alloc_obj_p << std::endl;
        return;           // Just skip unknown allocators
      }

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
        m_operation_mgr.bld_allocate(allocator_number, alloc_size);
      }
      else {
        const std::string memory_str = m_json["result"]["memory_ptr"];
        const uint64_t memory_ptr = std::stoul(memory_str, nullptr, 0);

        m_operation_mgr.bld_allocate_cont(memory_ptr);
      }
    }

    void replay_deallocate( void )
    {
      const std::string alloc_obj_s = m_json["payload"]["allocator_ref"];
      const uint64_t alloc_obj_p = std::stoul(alloc_obj_s, nullptr, 0);
      auto n_iter = m_allocator_indices.find(alloc_obj_p);

      if ( n_iter == m_allocator_indices.end() ) {
        std::cout << "Unable to find allocator for: " << m_json["payload"]["memory_ptr"] << " deallocation ignored" <<  std::endl;
        return;           // Just skip unknown allocators
      }

      const AllocatorIndex& allocator_number = n_iter->second;

      const std::string memory_str = m_json["payload"]["memory_ptr"];
      const uint64_t memory_ptr = std::stoul(memory_str, nullptr, 0);

      m_operation_mgr.bld_deallocate(allocator_number, memory_ptr);
    }

    void replay_coalesce( void )
    {
      std::string allocator_name = m_json["payload"]["allocator_name"];
      strip_off_base(allocator_name);

      m_operation_mgr.bld_coalesce(allocator_name);
    }

    void replay_release( void )
    {
      const std::string alloc_obj_s = m_json["payload"]["allocator_ref"];
      const uint64_t alloc_obj_p = std::stoul(alloc_obj_s, nullptr, 0);
      auto n_iter = m_allocator_indices.find(alloc_obj_p);

      if ( n_iter == m_allocator_indices.end() ) {
        std::cout << "Unable to find allocator for: " << m_json["payload"]["memory_ptr"] << " release ignored" <<  std::endl;
        return;
      }

      const AllocatorIndex& allocator_number = n_iter->second;
      m_operation_mgr.bld_release(allocator_number);
    }
};
#endif // REPLAY_Replay_HPP

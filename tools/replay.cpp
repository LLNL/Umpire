//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "umpire/config.hpp"

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/op/MemoryOperation.hpp"
#include "umpire/strategy/AllocationAdvisor.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/DynamicPoolHeuristic.hpp"
#include "umpire/strategy/FixedPool.hpp"
#include "umpire/strategy/MixedPool.hpp"
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/strategy/SlotPool.hpp"
#include "umpire/strategy/SizeLimiter.hpp"
#include "umpire/strategy/SlotPool.hpp"
#include "umpire/strategy/ThreadSafeAllocator.hpp"
#include "umpire/tpl/cxxopts/include/cxxopts.hpp"
#include "umpire/tpl/json/json.hpp"

static cxxopts::ParseResult parse(int argc, char* argv[])
{
  try
  {
    cxxopts::Options options(argv[0], "Replay an umpire session from a file");

    options
      .add_options()
      (  "h, help"
       , "Print help"
      )
      (  "i, infile"
       , "Input file created by Umpire library with UMPIRE_REPLAY=On"
       , cxxopts::value<std::string>(), "FILE"
      )
      (  "u, uid"
       , "The format of a REPLAY line begins with REPLAY,UID,...  This instructs replay to only replay items for the specified UID."
       , cxxopts::value<uint64_t>(), "UID"
      )
    ;

    options.add_options("HiddenGroup")
      (  "t, testfile"
       , "Generate a file to be used for unit testing."
       , cxxopts::value<std::string>(), "FILE"
      )
    ;

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
      // You can output our default, unnamed group and our HiddenGroup
      // of help with the following line:
      //
      //     std::cout << options.help({"", "HiddenGroup"}) << std::endl;
      //
      std::cout << options.help({""}) << std::endl;
      exit(0);
    }

    return result;
  } catch (const cxxopts::OptionException& e)
  {
    std::cout << "error parsing options: " << e.what() << std::endl;
    exit(1);
  }
}

class NullBuffer : public std::streambuf {
  public:
    int overflow(int c) { return c; }
};
static NullBuffer null_buffer;
static std::ostream null_stream(&null_buffer);

class Replay {
  public:
    Replay( cxxopts::ParseResult options, std::string in_file_name, std::string out_file_name ):
        m_sequence_id(0)
      , m_replay_uid(0)
      , m_options(options)
      , m_input_file(in_file_name)
      , m_rm(umpire::ResourceManager::getInstance())
    {
      if ( ! m_input_file.is_open() )
        usage_and_exit( "Unable to open input file " + in_file_name );

      if (out_file_name != "") {
        m_replayout.open(out_file_name);
        if ( ! m_replayout.is_open() )
          usage_and_exit( "Unable to open output file " + out_file_name );
        replay_out(m_replayout);
      }

      if ( m_options.count("uid") ) {
        m_replay_uid = m_options["uid"].as<uint64_t>();
        m_uids[m_replay_uid] = true;
      }
    }

    static void usage_and_exit( const std::string& /* errorMessage */ ) {
      exit (1);
    }

    void run(void)
    {
      while ( std::getline(m_input_file, m_line) ) {
        const std::string header("{ \"kind\":\"replay\", \"uid\":");
        auto header_len = header.size();

        if ( m_line.size() <= header_len || m_line.substr(0, header_len) != header.substr(0, header_len) )
          continue;

        m_json.clear();
        m_json = nlohmann::json::parse(m_line);

        if ( ! m_replay_uid ) {
          m_replay_uid = m_json["uid"];
          m_uids[m_json["uid"]] = true;
        }

        if ( m_replay_uid != m_json["uid"] ) {
          if ( m_uids.find(m_json["uid"]) == m_uids.end()) {
            //
            // Only note a message once per uid
            //
            std::cerr << "Skipping Replay for PID " << m_json["uid"] << std::endl;
            m_uids[m_json["uid"]] = true;
          }
          continue;
        }

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
          replay_out() << "deallocate ";
          replay_deallocate();
          replay_out() << std::endl;
        }
        else if ( m_json["event"] == "coalesce" ) {
          replay_out() << "coalesce ";
          replay_coalesce();
          replay_out() << std::endl;
        }
        else if ( m_json["event"] == "release" ) {
          replay_out() << "release ";
          replay_release();
          replay_out() << std::endl;
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
          replay_out() << m_json["event"] << " ";
          std::cerr << "Unknown Replay (" << m_json["event"] << ")\n";
          replay_out() << "\n";
          exit (1);
        }
      }
    }

  private:
    uint64_t m_sequence_id;
    uint64_t m_replay_uid;
    cxxopts::ParseResult m_options;
    std::ifstream m_input_file;
    umpire::ResourceManager& m_rm;
    std::ofstream m_replayout;
    std::unordered_map<uint64_t, bool> m_uids;  // key(uid), val(always true)
    std::unordered_map<std::string, std::string> m_allocators;  // key(alloc_obj), val(alloc name)
    std::unordered_map<std::string, void*> m_allocated_ptrs;    // key(alloc_ptr), val(replay_alloc_ptr)
    std::unordered_map<std::string, uint64_t> m_allocation_seq;    // key(alloc_ptr), val(m_sequence_id)
    std::string m_line;
    nlohmann::json m_json;
    std::vector<std::string> m_row;
    void* m_alloc_ptr;

    std::ostream& replay_out(std::ostream& outs = null_stream)
    {
      static std::ostream& rpl_out = outs;
      return rpl_out;
    }

    template <typename T>
    void get_from_string( const std::string& s, T& val )
    {
        std::istringstream ss(s);
        ss >> val;
    }

    void strip_off_base(std::string& s)
    {
      const std::string base("_base");

      if (s.length() > base.length()) {
        if (s.compare(s.length() - base.length(), base.length(), base) == 0) {
          s.erase(s.length() - base.length(), base.length());
        }
      }
    }

    void replay_coalesce( void )
    {
      std::string allocator_name = m_json["payload"]["allocator_name"];
      strip_off_base(allocator_name);

      replay_out() << allocator_name;

      try {
        auto alloc = m_rm.getAllocator(allocator_name);
        auto strategy = alloc.getAllocationStrategy();
        auto tracker = dynamic_cast<umpire::strategy::AllocationTracker*>(strategy);

        if (tracker)
          strategy = tracker->getAllocationStrategy();

        auto dynamic_pool = dynamic_cast<umpire::strategy::DynamicPool*>(strategy);

        if (dynamic_pool) {
          dynamic_pool->coalesce();
        }
        else {
          std::cerr << allocator_name << " is not a dynamic pool, skipping\n";
          return;
        }
      }
      catch (std::exception& e) {
        std::cerr << "Unable to find allocator for " << allocator_name << '\n'
          << e.what() << '\n'
          << "Skipped\n";
        return;
      }
    }

    void replay_release( void )
    {
      auto n_iter = m_allocators.find(m_json["payload"]["allocator_ref"]);

      if ( n_iter == m_allocators.end() ) {
        std::cerr << "Unknown allocator " << m_json["payload"]["allocator_ref"] << std::endl;
        return;           // Just skip unknown allocators
      }

      const std::string& allocName = n_iter->second;

      replay_out() << allocName;

      auto alloc = m_rm.getAllocator(allocName);
      alloc.release();
    }

    void replay_allocate( void )
    {
      std::size_t alloc_size = m_json["payload"]["size"];

      auto n_iter = m_allocators.find(m_json["payload"]["allocator_ref"]);

      if ( n_iter == m_allocators.end() ) {
        std::cerr << "Unknown allocator " << m_json["payload"]["allocator_ref"] << std::endl;
        return;           // Just skip unknown allocators
      }

      const std::string& allocName = n_iter->second;

      if ( m_json["result"].is_null() ) {
        auto alloc = m_rm.getAllocator(allocName);

        m_alloc_ptr = alloc.allocate(alloc_size);
      }
      else {
        m_allocated_ptrs[m_json["result"]["memory_ptr"]] = m_alloc_ptr;
        m_sequence_id++;
        m_allocation_seq[m_json["result"]["memory_ptr"]] = m_sequence_id;

        replay_out()
          << "allocate "
          << "(" << alloc_size << ") " << allocName << " --> " << m_sequence_id
          << std::endl;
      }
    }

    void replay_deallocate( void )
    {
      auto n_iter = m_allocators.find(m_json["payload"]["allocator_ref"]);

      if ( n_iter == m_allocators.end() ) {
        std::cout << "Unable to find allocator for: " << m_json["payload"]["memory_ptr"] << " deallocation ignored" <<  std::endl;
        return;           // Just skip unknown allocators
      }

      const std::string& allocName = n_iter->second;

      replay_out() << allocName;

      auto p_iter = m_allocated_ptrs.find(m_json["payload"]["memory_ptr"]);
      if ( p_iter == m_allocated_ptrs.end() ) {
        std::cout << "Duplicate deallocate for:" << m_json["payload"]["memory_ptr"] << " ignored" <<  std::endl;
        return;           // Just skip unknown allocators
      }

      auto s_iter = m_allocation_seq.find(m_json["payload"]["memory_ptr"]);
      replay_out() << "(" << s_iter->second << ")";
      m_allocation_seq.erase(s_iter);

      void* replay_alloc_ptr = p_iter->second;
      m_allocated_ptrs.erase(p_iter);

      auto alloc = m_rm.getAllocator(allocName);
      alloc.deallocate(replay_alloc_ptr);
    }

    void replay_makeMemoryResource( void )
    {
      m_allocators[ m_json["result"] ] = m_json["payload"]["name"];
    }

    void replay_makeAllocator( void )
    {
      // std::cout << m_json.dump() << std::endl;
      const std::string& allocator_name = m_json["payload"]["allocator_name"];

      //
      // When the result isn't set, just perform the operation.  We will
      // establish the mapping on after the result has been recorded in the
      // two-step REPLAY process for this event.
      //
      if ( m_json["result"].is_null() ) {
        replay_out() << "makeAllocator ";
        bool introspection = m_json["payload"]["with_introspection"];

        std::string type = m_json["payload"]["type"];

        replay_out() << "<" << type << ">" ;

        if ( type == "umpire::strategy::AllocationAdvisor" ) {
          const std::string& base_allocator_name = m_json["payload"]["args"][0];
          const std::string& advice_operation = m_json["payload"]["args"][1];

          // Now grab the optional fields
          if (m_json["payload"]["args"].size() >= 3) {
            const std::string& accessing_allocator_name = m_json["payload"]["args"][2];

            replay_out()
              << "(" << allocator_name
              << ", getAllocator(" << base_allocator_name << ")"
              << ", " << advice_operation
              << ", getAllocator(" << accessing_allocator_name << ")"
              << ")";

            if ( introspection )  m_rm.makeAllocator<umpire::strategy::AllocationAdvisor, true>( allocator_name, m_rm.getAllocator(base_allocator_name), advice_operation, m_rm.getAllocator(accessing_allocator_name));
            else                  m_rm.makeAllocator<umpire::strategy::AllocationAdvisor, false>(allocator_name, m_rm.getAllocator(base_allocator_name), advice_operation, m_rm.getAllocator(accessing_allocator_name));
          }
          else {

            replay_out()
              << "(" << allocator_name
              << ", getAllocator(" << base_allocator_name << ")"
              << ", " << advice_operation
              << ")";

            if ( introspection )  m_rm.makeAllocator<umpire::strategy::AllocationAdvisor, true>( allocator_name, m_rm.getAllocator(base_allocator_name), advice_operation);
            else                  m_rm.makeAllocator<umpire::strategy::AllocationAdvisor, false>(allocator_name, m_rm.getAllocator(base_allocator_name), advice_operation);
          }
        }
        else if ( type == "umpire::strategy::DynamicPool" ) {
          const std::string& base_allocator_name = m_json["payload"]["args"][0];

          std::size_t initial_alloc_size;
          std::size_t min_alloc_size;
          int alignment;

          // Now grab the optional fields
          if (m_json["payload"]["args"].size() >= 4) {
            get_from_string(m_json["payload"]["args"][1], initial_alloc_size);
            get_from_string(m_json["payload"]["args"][2], min_alloc_size);
            get_from_string(m_json["payload"]["args"][3], alignment);

            replay_out()
              << "(" << allocator_name
              << ", getAllocator(" << base_allocator_name << ")"
              << ", " << initial_alloc_size
              << ", " << min_alloc_size
              << ", " << alignment
              << ")";

            // NOTE Replay cannot handle user heuristics, so set to umpire::strategy::heuristic_percent_releasable(0) to turn it off
            if ( introspection )  m_rm.makeAllocator<umpire::strategy::DynamicPool, true>(allocator_name, m_rm.getAllocator(base_allocator_name), initial_alloc_size, min_alloc_size, alignment, umpire::strategy::heuristic_percent_releasable(0));
            else                  m_rm.makeAllocator<umpire::strategy::DynamicPool, false>(allocator_name, m_rm.getAllocator(base_allocator_name), initial_alloc_size, min_alloc_size, alignment, umpire::strategy::heuristic_percent_releasable(0));
          }
          else if (m_json["payload"]["args"].size() >= 3) {
            get_from_string(m_json["payload"]["args"][1], initial_alloc_size);
            get_from_string(m_json["payload"]["args"][2], min_alloc_size);

            replay_out()
              << "(" << allocator_name
              << ", getAllocator(" << base_allocator_name << ")"
              << ", " << initial_alloc_size
              << ", " << min_alloc_size
              << ")";

            if ( introspection )  m_rm.makeAllocator<umpire::strategy::DynamicPool, true>(allocator_name, m_rm.getAllocator(base_allocator_name), initial_alloc_size, min_alloc_size);
            else                  m_rm.makeAllocator<umpire::strategy::DynamicPool, false>(allocator_name, m_rm.getAllocator(base_allocator_name), initial_alloc_size, min_alloc_size);
          }
          else if (m_json["payload"]["args"].size() >= 2) {
            get_from_string(m_json["payload"]["args"][1], initial_alloc_size);

            replay_out()
              << "(" << allocator_name
              << ", getAllocator(" << base_allocator_name << ")"
              << ", " << initial_alloc_size
              << ")";

            if ( introspection )  m_rm.makeAllocator<umpire::strategy::DynamicPool, true>(allocator_name, m_rm.getAllocator(base_allocator_name), initial_alloc_size);
            else                  m_rm.makeAllocator<umpire::strategy::DynamicPool, false>(allocator_name, m_rm.getAllocator(base_allocator_name), initial_alloc_size);
          }
          else {
            replay_out()
              << "(" << allocator_name
              << ", getAllocator(" << base_allocator_name << ")"
              << ")";

            if ( introspection )  m_rm.makeAllocator<umpire::strategy::DynamicPool, true>(allocator_name, m_rm.getAllocator(base_allocator_name));
            else                  m_rm.makeAllocator<umpire::strategy::DynamicPool, false>(allocator_name, m_rm.getAllocator(base_allocator_name));
          }
        }
        else if ( type == "umpire::strategy::MonotonicAllocationStrategy" ) {
          const std::string& base_allocator_name = m_json["payload"]["args"][1];

          std::size_t capacity;
          get_from_string(m_json["payload"]["args"][0], capacity);

          replay_out()
            << "(" << allocator_name
            << ", " << capacity
              << ", getAllocator(" << base_allocator_name << ")"
            << ")";

          if ( introspection )  m_rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy, true>(allocator_name, capacity, m_rm.getAllocator(base_allocator_name));
          else                  m_rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy, false>(allocator_name, capacity, m_rm.getAllocator(base_allocator_name));
        }
        else if ( type == "umpire::strategy::SlotPool" ) {
          const std::string& base_allocator_name = m_json["payload"]["args"][1];

          std::size_t slots;
          get_from_string(m_json["payload"]["args"][0], slots);

          replay_out()
            << "(" << allocator_name
            << ", " << slots
            << ", getAllocator(" << base_allocator_name << ")"
            << ")";

          if ( introspection )  m_rm.makeAllocator<umpire::strategy::SlotPool, true>(allocator_name, slots, m_rm.getAllocator(base_allocator_name));
          else                  m_rm.makeAllocator<umpire::strategy::SlotPool, false>(allocator_name, slots, m_rm.getAllocator(base_allocator_name));
        }
        else if ( type == "umpire::strategy::SizeLimiter" ) {
          const std::string& base_allocator_name = m_json["payload"]["args"][0];
          std::size_t size_limit;
          get_from_string(m_json["payload"]["args"][1], size_limit);

          replay_out()
            << "(" << allocator_name
            << ", getAllocator(" << base_allocator_name << ")"
            << ", " << size_limit
            << ")";

          if ( introspection )  m_rm.makeAllocator<umpire::strategy::SizeLimiter, true>(allocator_name, m_rm.getAllocator(base_allocator_name), size_limit);
          else                  m_rm.makeAllocator<umpire::strategy::SizeLimiter, false>(allocator_name, m_rm.getAllocator(base_allocator_name), size_limit);
        }
        else if ( type == "umpire::strategy::ThreadSafeAllocator" ) {
          const std::string& base_allocator_name = m_json["payload"]["args"][0];

          replay_out()
            << "(" << allocator_name
            << ", getAllocator(" << base_allocator_name << ")"
            << ")";

          if ( introspection )  m_rm.makeAllocator<umpire::strategy::ThreadSafeAllocator, true>(allocator_name, m_rm.getAllocator(base_allocator_name));
          else                  m_rm.makeAllocator<umpire::strategy::ThreadSafeAllocator, false>(allocator_name, m_rm.getAllocator(base_allocator_name));
        }
        else if ( type == "umpire::strategy::FixedPool" ) {
          //
          // Need to skip FixedPool for now since I haven't figured out how to
          // dynamically parse/creat the data type parameter
          //
          replay_out() << " (ignored) ";
          return;
        }
        else if ( type == "umpire::strategy::MixedPool" ) {
          const std::string& base_allocator_name = m_json["payload"]["args"][0];
          std::size_t smallest_fixed_blocksize;
          std::size_t largest_fixed_blocksize;
          std::size_t max_fixed_blocksize;
          std::size_t size_multiplier;
          std::size_t dynamic_initial_alloc_bytes;
          std::size_t dynamic_min_alloc_bytes;
          int dynamic_align_bytes = 16;

          // Now grab the optional fields
          if (m_json["payload"]["args"].size() >= 8) {
            get_from_string(m_json["payload"]["args"][1], smallest_fixed_blocksize);
            get_from_string(m_json["payload"]["args"][2], largest_fixed_blocksize);
            get_from_string(m_json["payload"]["args"][3], max_fixed_blocksize);
            get_from_string(m_json["payload"]["args"][4], size_multiplier);
            get_from_string(m_json["payload"]["args"][5], dynamic_initial_alloc_bytes);
            get_from_string(m_json["payload"]["args"][6], dynamic_min_alloc_bytes);
            get_from_string(m_json["payload"]["args"][7], dynamic_align_bytes);

            replay_out()
              << "(" << allocator_name
              << ", getAllocator(" << base_allocator_name << ")"
              << ", " << smallest_fixed_blocksize
              << ", " << largest_fixed_blocksize
              << ", " << max_fixed_blocksize
              << ", " << size_multiplier
              << ", " << dynamic_initial_alloc_bytes
              << ", " << dynamic_min_alloc_bytes
              << ", " << dynamic_align_bytes
              << ")";

            if ( introspection )
              m_rm.makeAllocator<umpire::strategy::MixedPool, true>
                   (allocator_name, m_rm.getAllocator(base_allocator_name)
                     , smallest_fixed_blocksize
                     , largest_fixed_blocksize
                     , max_fixed_blocksize
                     , size_multiplier
                     , dynamic_initial_alloc_bytes
                     , dynamic_min_alloc_bytes
                     , dynamic_align_bytes
                     , umpire::strategy::heuristic_percent_releasable(0)
                 );
            else
              m_rm.makeAllocator<umpire::strategy::MixedPool, false>
                   (allocator_name, m_rm.getAllocator(base_allocator_name)
                     , smallest_fixed_blocksize
                     , largest_fixed_blocksize
                     , max_fixed_blocksize
                     , size_multiplier
                     , dynamic_initial_alloc_bytes
                     , dynamic_min_alloc_bytes
                     , dynamic_align_bytes
                     , umpire::strategy::heuristic_percent_releasable(0)
                 );
          }
          else if (m_json["payload"]["args"].size() >= 7) {
            get_from_string(m_json["payload"]["args"][1], smallest_fixed_blocksize);
            get_from_string(m_json["payload"]["args"][2], largest_fixed_blocksize);
            get_from_string(m_json["payload"]["args"][3], max_fixed_blocksize);
            get_from_string(m_json["payload"]["args"][4], size_multiplier);
            get_from_string(m_json["payload"]["args"][5], dynamic_initial_alloc_bytes);
            get_from_string(m_json["payload"]["args"][6], dynamic_min_alloc_bytes);

            replay_out()
              << "(" << allocator_name
              << ", getAllocator(" << base_allocator_name << ")"
              << ", " << smallest_fixed_blocksize
              << ", " << largest_fixed_blocksize
              << ", " << max_fixed_blocksize
              << ", " << size_multiplier
              << ", " << dynamic_initial_alloc_bytes
              << ", " << dynamic_min_alloc_bytes
              << ")";

            if ( introspection )
              m_rm.makeAllocator<umpire::strategy::MixedPool, true>
                   (allocator_name, m_rm.getAllocator(base_allocator_name)
                     , smallest_fixed_blocksize
                     , largest_fixed_blocksize
                     , max_fixed_blocksize
                     , size_multiplier
                     , dynamic_initial_alloc_bytes
                     , dynamic_min_alloc_bytes
                 );
            else
              m_rm.makeAllocator<umpire::strategy::MixedPool, false>
                   (allocator_name, m_rm.getAllocator(base_allocator_name)
                     , smallest_fixed_blocksize
                     , largest_fixed_blocksize
                     , max_fixed_blocksize
                     , size_multiplier
                     , dynamic_initial_alloc_bytes
                     , dynamic_min_alloc_bytes
                 );
          }
          else if (m_json["payload"]["args"].size() >= 6) {
            get_from_string(m_json["payload"]["args"][1], smallest_fixed_blocksize);
            get_from_string(m_json["payload"]["args"][2], largest_fixed_blocksize);
            get_from_string(m_json["payload"]["args"][3], max_fixed_blocksize);
            get_from_string(m_json["payload"]["args"][4], size_multiplier);
            get_from_string(m_json["payload"]["args"][5], dynamic_initial_alloc_bytes);

            replay_out()
              << "(" << allocator_name
              << ", getAllocator(" << base_allocator_name << ")"
              << ", " << smallest_fixed_blocksize
              << ", " << largest_fixed_blocksize
              << ", " << max_fixed_blocksize
              << ", " << size_multiplier
              << ", " << dynamic_initial_alloc_bytes
              << ")";

            if ( introspection )
              m_rm.makeAllocator<umpire::strategy::MixedPool, true>
                   (allocator_name, m_rm.getAllocator(base_allocator_name)
                     , smallest_fixed_blocksize
                     , largest_fixed_blocksize
                     , max_fixed_blocksize
                     , size_multiplier
                     , dynamic_initial_alloc_bytes
                 );
            else
              m_rm.makeAllocator<umpire::strategy::MixedPool, false>
                   (allocator_name, m_rm.getAllocator(base_allocator_name)
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

            replay_out()
              << "(" << allocator_name
              << ", getAllocator(" << base_allocator_name << ")"
              << ", " << smallest_fixed_blocksize
              << ", " << largest_fixed_blocksize
              << ", " << max_fixed_blocksize
              << ", " << size_multiplier
              << ")";

            if ( introspection )
              m_rm.makeAllocator<umpire::strategy::MixedPool, true>
                   (allocator_name, m_rm.getAllocator(base_allocator_name)
                     , smallest_fixed_blocksize
                     , largest_fixed_blocksize
                     , max_fixed_blocksize
                     , size_multiplier
                 );
            else
              m_rm.makeAllocator<umpire::strategy::MixedPool, false>
                   (allocator_name, m_rm.getAllocator(base_allocator_name)
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

            replay_out()
              << "(" << allocator_name
              << ", getAllocator(" << base_allocator_name << ")"
              << ", " << smallest_fixed_blocksize
              << ", " << largest_fixed_blocksize
              << ", " << max_fixed_blocksize
              << ")";

            if ( introspection )
              m_rm.makeAllocator<umpire::strategy::MixedPool, true>
                   (allocator_name, m_rm.getAllocator(base_allocator_name)
                     , smallest_fixed_blocksize
                     , largest_fixed_blocksize
                     , max_fixed_blocksize
                 );
            else
              m_rm.makeAllocator<umpire::strategy::MixedPool, false>
                   (allocator_name, m_rm.getAllocator(base_allocator_name)
                     , smallest_fixed_blocksize
                     , largest_fixed_blocksize
                     , max_fixed_blocksize
                 );
          }
          else if (m_json["payload"]["args"].size() >= 3) {
            get_from_string(m_json["payload"]["args"][1], smallest_fixed_blocksize);
            get_from_string(m_json["payload"]["args"][2], largest_fixed_blocksize);

            replay_out()
              << "(" << allocator_name
              << ", getAllocator(" << base_allocator_name << ")"
              << ", " << smallest_fixed_blocksize
              << ", " << largest_fixed_blocksize
              << ")";

            if ( introspection )
              m_rm.makeAllocator<umpire::strategy::MixedPool, true>
                   (allocator_name, m_rm.getAllocator(base_allocator_name)
                     , smallest_fixed_blocksize
                     , largest_fixed_blocksize
                 );
            else
              m_rm.makeAllocator<umpire::strategy::MixedPool, false>
                   (allocator_name, m_rm.getAllocator(base_allocator_name)
                     , smallest_fixed_blocksize
                     , largest_fixed_blocksize
                 );
          }
          else if (m_json["payload"]["args"].size() >= 2) {
            get_from_string(m_json["payload"]["args"][1], smallest_fixed_blocksize);

            replay_out()
              << "(" << allocator_name
              << ", getAllocator(" << base_allocator_name << ")"
              << ", " << smallest_fixed_blocksize
              << ")";

            if ( introspection )
              m_rm.makeAllocator<umpire::strategy::MixedPool, true>
                   (allocator_name, m_rm.getAllocator(base_allocator_name)
                     , smallest_fixed_blocksize
                 );
            else
              m_rm.makeAllocator<umpire::strategy::MixedPool, false>
                   (allocator_name, m_rm.getAllocator(base_allocator_name)
                     , smallest_fixed_blocksize
                 );
          }
          else {
            replay_out()
              << "(" << allocator_name
              << ", getAllocator(" << base_allocator_name << ")"
              << ")";

            if ( introspection )
              m_rm.makeAllocator<umpire::strategy::MixedPool, true>
                   (allocator_name, m_rm.getAllocator(base_allocator_name)
                 );
            else
              m_rm.makeAllocator<umpire::strategy::MixedPool, false>
                   (allocator_name, m_rm.getAllocator(base_allocator_name)
                 );
          }
        }
        else {
          std::cerr << "Unknown class (" << type << "), skipping.\n";
          return;
        }
        replay_out() << "\n";
      }
      else {
        m_allocators[ m_json["result"]["allocator_ref"] ] = allocator_name;
      }
    }
};


int main(int ac, char* av[])
{
  auto result = parse(ac, av);

  if ( ! result.count("infile") ) {
    std::cerr << "No input file specified\n";
    exit(1);
  }

  std::string input_file_name = result["infile"].as<std::string>();

  std::string output_file_name;
  if ( result.count("testfile") )
    output_file_name = result["testfile"].as<std::string>();

  Replay replay(result, input_file_name, output_file_name);

  replay.run();

  return 0;
}

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
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/op/MemoryOperation.hpp"
#include "umpire/strategy/AllocationAdvisor.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/DynamicPoolHeuristic.hpp"
#include "umpire/strategy/FixedPool.hpp"
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/strategy/SlotPool.hpp"
#include "umpire/strategy/SizeLimiter.hpp"
#include "umpire/strategy/ThreadSafeAllocator.hpp"

class CSVRow {
public:
  std::string const& operator[](std::size_t index) const { return m_data[index]; }
  std::size_t size() const { return m_data.size(); }
  void readNextRow(std::istream& str) {
    std::string line;
    std::getline(str, line);

    std::stringstream lineStream(line);
    std::string cell;

    m_data.clear();
    while ( std::getline(lineStream, cell, ',') ) {
      m_data.push_back(cell);
    }

    // This checks for a trailing comma with no data after it.
    if (!lineStream && cell.empty()) {
      // If there was a trailing comma then add an empty element.
      m_data.push_back("");
    }
  }
private:
  std::vector<std::string>    m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data)
{
  data.readNextRow(str);
  return str;
}   

class Replay {
  public:
    // Replay( const std::string& filename ) :
    Replay( const char* _fname ) :
       m_filename(_fname)
      , m_file(m_filename)
      , m_rm(umpire::ResourceManager::getInstance())
    {
      if ( ! m_file.is_open() )
        usage_and_exit( "Unable to open file " + m_filename );
    }

    static void usage_and_exit( const std::string& errorMessage ) {
      std::cerr << errorMessage
      << std::endl
      << "Usage: replay <replayfile.csv>"
      << std::endl;
      exit (1);
    }

    void run(void)
    {
      while ( m_file >> m_row ) {
        if ( m_row[0] != "REPLAY" )
          continue;

        if ( m_row[1] == "makeAllocator" ) {
          replay_makeAllocator();
        }
        else if ( m_row[1] == "makeMemoryResource" ) {
          replay_makeMemoryResource();
        }
        else if ( m_row[1] == "allocate" ) {
          replay_allocate();
        }
        else if ( m_row[1] == "deallocate" ) {
          replay_deallocate();
        }
        else if ( m_row[1] == "coalesce" ) {
          replay_coalesce();
        }
        else if ( m_row[1] == "release" ) {
          replay_release();
        }
        else {
          std::cout << m_row[1] << "\n";
        }
      }
    }

  private:
    std::string m_filename;
    std::ifstream m_file;
    umpire::ResourceManager& m_rm;
    std::unordered_map<void*, std::string> m_allocators;  // key(alloc_obj), val(alloc name)
    std::unordered_map<void*, void*> m_allocated_ptrs;    // key(alloc_ptr), val(replay_alloc_ptr)
    CSVRow m_row;

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
      std::string allocName = m_row[m_row.size() - 1];
      strip_off_base(allocName);

      try {
        auto alloc = m_rm.getAllocator(allocName);
        auto strategy = alloc.getAllocationStrategy();
        auto tracker = dynamic_cast<umpire::strategy::AllocationTracker*>(strategy);

        if (tracker)
          strategy = tracker->getAllocationStrategy();

        auto dynamic_pool = dynamic_cast<umpire::strategy::DynamicPool*>(strategy);

        if (dynamic_pool) {
          dynamic_pool->coalesce();
        }
        else {
          std::cerr << allocName << " is not a dynamic pool, skipping\n";
          return;
        }
      }
      catch (std::exception& e) {
        std::cerr << "Unable to find allocator for " << allocName << '\n'
          << e.what() << '\n'
          << "Skipped\n";
        return;
      }
    }

    void replay_release( void )
    {
      void* alloc_obj_ref;

      get_from_string(m_row[m_row.size() - 1], alloc_obj_ref);

      auto n_iter = m_allocators.find(alloc_obj_ref);

      if ( n_iter == m_allocators.end() ) {
        std::cerr << "Unknown allocator " << alloc_obj_ref << std::endl;
        return;           // Just skip unknown allocators
      }

      const std::string& allocName = n_iter->second;

      auto alloc = m_rm.getAllocator(allocName);
      alloc.release();
    }

    void replay_allocate( void )
    {
      void* alloc_obj_ref;
      std::size_t alloc_size;
      void* alloc_ptr;
      void* replay_alloc_ptr;

      get_from_string(m_row[m_row.size() - 1], alloc_ptr);
      get_from_string(m_row[m_row.size() - 2], alloc_obj_ref);
      get_from_string(m_row[m_row.size() - 3], alloc_size);

      auto n_iter = m_allocators.find(alloc_obj_ref);

      if ( n_iter == m_allocators.end() ) {
        std::cerr << "Unknown allocator " << alloc_obj_ref << std::endl;
        return;           // Just skip unknown allocators
      }

      const std::string& allocName = n_iter->second;

      auto alloc = m_rm.getAllocator(allocName);
      replay_alloc_ptr = alloc.allocate(alloc_size);

      m_allocated_ptrs[alloc_ptr] = replay_alloc_ptr;
    }

    void replay_deallocate( void )
    {
      void* alloc_obj_ref;
      void* alloc_ptr;

      get_from_string(m_row[m_row.size() - 1], alloc_obj_ref);
      get_from_string(m_row[m_row.size() - 2], alloc_ptr);

      auto n_iter = m_allocators.find(alloc_obj_ref);

      if ( n_iter == m_allocators.end() ) {
        std::cout << "Unable to find allocator for: " << alloc_ptr << " deallocation ignored" <<  std::endl;
        return;           // Just skip unknown allocators
      }

      auto p_iter = m_allocated_ptrs.find(alloc_ptr);
      if ( p_iter == m_allocated_ptrs.end() ) {
        std::cout << "Duplicate deallocate for:" << alloc_ptr << " ignored" <<  std::endl;
        return;           // Just skip unknown allocators
      }

      void* replay_alloc_ptr = p_iter->second;
      m_allocated_ptrs.erase(p_iter);

      const std::string& allocName = n_iter->second;

      auto alloc = m_rm.getAllocator(allocName);
      alloc.deallocate(replay_alloc_ptr);
    }

    void replay_makeMemoryResource( void )
    {
      void* alloc_obj_ref;

      const std::string& name = m_row[2];
      get_from_string(m_row[3], alloc_obj_ref);

      m_allocators[alloc_obj_ref] = name;
    }

    void replay_makeAllocator( void )
    {
      bool introspection = ( m_row[3] == "true" );
      const std::string& name = m_row[4];
      void* alloc_obj_ref;

      get_from_string(m_row[m_row.size() - 1], alloc_obj_ref);

      if ( m_row[2] == "umpire::strategy::AllocationAdvisor" ) {
        const std::string& allocName = m_row[5];
        const std::string& adviceOperation = m_row[6];
        // Now grab the optional fields
        if (m_row.size() > 8) {
          const std::string& accessingAllocatorName = m_row[7];
          if ( introspection )  m_rm.makeAllocator<umpire::strategy::AllocationAdvisor, true>( name, m_rm.getAllocator(allocName), adviceOperation, m_rm.getAllocator(accessingAllocatorName));
          else                  m_rm.makeAllocator<umpire::strategy::AllocationAdvisor, false>(name, m_rm.getAllocator(allocName), adviceOperation, m_rm.getAllocator(accessingAllocatorName));
        }
        else {
          if ( introspection )  m_rm.makeAllocator<umpire::strategy::AllocationAdvisor, true>( name, m_rm.getAllocator(allocName), adviceOperation);
          else                  m_rm.makeAllocator<umpire::strategy::AllocationAdvisor, false>(name, m_rm.getAllocator(allocName), adviceOperation);
        }
      }
      else if ( m_row[2] == "umpire::strategy::DynamicPool" ) {
        const std::string& allocName = m_row[5];
        std::size_t min_initial_alloc_size; // Optional: m_row[6]
        std::size_t min_alloc_size;         // Optional: m_row[7]

        // Now grab the optional fields
        if (m_row.size() > 8) {
          get_from_string(m_row[6], min_initial_alloc_size);
          get_from_string(m_row[7], min_alloc_size);
          if ( introspection )  m_rm.makeAllocator<umpire::strategy::DynamicPool, true>(name, m_rm.getAllocator(allocName), min_initial_alloc_size, min_alloc_size, umpire::strategy::heuristic_percent_releasable(0));
          else                  m_rm.makeAllocator<umpire::strategy::DynamicPool, false>(name, m_rm.getAllocator(allocName), min_initial_alloc_size, min_alloc_size, umpire::strategy::heuristic_percent_releasable(0));
        }
        else if ( m_row.size() > 7 ) {
          get_from_string(m_row[6], min_initial_alloc_size);
          if ( introspection )  m_rm.makeAllocator<umpire::strategy::DynamicPool, true>(name, m_rm.getAllocator(allocName), min_initial_alloc_size);
          else                  m_rm.makeAllocator<umpire::strategy::DynamicPool, false>(name, m_rm.getAllocator(allocName), min_initial_alloc_size);
        }
        else {
          if ( introspection )  m_rm.makeAllocator<umpire::strategy::DynamicPool, true>(name, m_rm.getAllocator(allocName));
          else                  m_rm.makeAllocator<umpire::strategy::DynamicPool, false>(name, m_rm.getAllocator(allocName));
        }
      }
      else if ( m_row[2] == "umpire::strategy::MonotonicAllocationStrategy" ) {
        std::size_t capacity;
        get_from_string(m_row[5], capacity);

        const std::string& allocName = m_row[6];

        if ( introspection )  m_rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy, true>(name, capacity, m_rm.getAllocator(allocName));
        else                  m_rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy, false>(name, capacity, m_rm.getAllocator(allocName));
      }
      else if ( m_row[2] == "umpire::strategy::SizeLimiter" ) {
        const std::string& allocName = m_row[5];
        std::size_t size_limit;
        get_from_string(m_row[6], size_limit);

        if ( introspection )  m_rm.makeAllocator<umpire::strategy::SizeLimiter, true>(name, m_rm.getAllocator(allocName), size_limit);
        else                  m_rm.makeAllocator<umpire::strategy::SizeLimiter, false>(name, m_rm.getAllocator(allocName), size_limit);
      }
      else if ( m_row[2] == "umpire::strategy::SlotPool" ) {
        const std::string& allocName = m_row[6];
        std::size_t slots;
        get_from_string(m_row[5], slots);

        if ( introspection )  m_rm.makeAllocator<umpire::strategy::SlotPool, true>(name, slots, m_rm.getAllocator(allocName));
        else                  m_rm.makeAllocator<umpire::strategy::SlotPool, false>(name, slots, m_rm.getAllocator(allocName));
      }
      else if ( m_row[2] == "umpire::strategy::ThreadSafeAllocator" ) {
        const std::string& allocName = m_row[5];

        if ( introspection )  m_rm.makeAllocator<umpire::strategy::ThreadSafeAllocator, true>(name, m_rm.getAllocator(allocName));
        else                  m_rm.makeAllocator<umpire::strategy::ThreadSafeAllocator, false>(name, m_rm.getAllocator(allocName));
      }
      else if ( m_row[2] == "umpire::strategy::FixedPool" ) {
        //
        // Need to skip FixedPool for now since I haven't figured out how to
        // dynamically parse/creat the data type parameter
        //
        return;
#if 0
        const std::string& allocName = m_row[5];
        std::size_t PoolSize = hmm...

        if ( introspection )  m_rm.makeAllocator<umpire::strategy::FixedPool<PoolSize>, true>(name, m_rm.getAllocator(allocName));
        else                  m_rm.makeAllocator<umpire::strategy::FixedPool<PoolSize>, false>(name, m_rm.getAllocator(allocName));
#endif
      }
      else {
        std::cerr << "Unknown class (" << m_row[2] << "), skipping.\n";
        return;
      }

      m_allocators[alloc_obj_ref] = name;
    }
};

int main(int ac, char** av)
{
  if ( ac != 2 )
    Replay::usage_and_exit( "Incorrect number of program arguments" );

  Replay replay(av[1]);

  replay.run();

  return 0;
}

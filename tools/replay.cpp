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
#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <sstream>
#include <vector>
#include <unordered_map>

#include <cstdlib>

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/SlotPool.hpp"
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/op/MemoryOperation.hpp"
#include "umpire/strategy/AllocationAdvisor.hpp"

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
        else {
          std::cout << m_row[1] << "\n";
        }
      }
    }

  private:
    std::string m_filename;
    std::ifstream m_file;
    umpire::ResourceManager& m_rm;
    std::unordered_map<void*, std::string> m_allocators;
    CSVRow m_row;

    template <typename T>
    void get_from_string( const std::string& s, T& val )
    {
        std::istringstream ss(s);
        ss >> val;
    }

    void replay_makeAllocator( void )
    {
      bool introspection = ( m_row[3] == "true" );
      void* alloc_obj_ref;

      get_from_string(m_row[m_row.size() - 1], alloc_obj_ref);

      if ( m_row[2] == "umpire::strategy::AllocationAdvisor" ) {
        const std::string& name = m_row[4];
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
        const std::string& name = m_row[4];
        const std::string& allocName = m_row[5];
        std::size_t min_initial_alloc_size; // Optional: m_row[6]
        std::size_t min_alloc_size;         // Optional: m_row[7]

        // Now grab the optional fields
        if (m_row.size() > 8) {
          get_from_string(m_row[6], min_initial_alloc_size);
          get_from_string(m_row[7], min_alloc_size);
          if ( introspection )  m_rm.makeAllocator<umpire::strategy::DynamicPool, true>(name, m_rm.getAllocator(allocName), min_initial_alloc_size, min_alloc_size);
          else                  m_rm.makeAllocator<umpire::strategy::DynamicPool, false>(name, m_rm.getAllocator(allocName), min_initial_alloc_size, min_alloc_size);
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

        m_allocators[alloc_obj_ref] = name;
      }
      else if ( m_row[2] == "umpire::strategy::MonotonicAllocationStrategy" ) {
      }
      else if ( m_row[2] == "umpire::strategy::SizeLimiter" ) {
      }
      else if ( m_row[2] == "umpire::strategy::SlotPool" ) {
      }
      else if ( m_row[2] == "umpire::strategy::ThreadSafeAllocator" ) {
      }
      else if ( m_row[2] == "umpire::strategy::FixedPool<data>" ) {
      }
      else {
        std::cerr << "Unknown class (" << m_row[2] << "), skipping.\n";
      }
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

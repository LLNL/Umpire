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
#include "umpire/strategy/SizeLimiter.hpp"
#include "umpire/strategy/ThreadSafeAllocator.hpp"
#include "umpire/strategy/FixedPool.hpp"
#include "umpire/util/AllocationMap.hpp"
#include "umpire/util/AllocationRecord.hpp"

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
    Replay( const char* _fname ) :
        m_filename(_fname)
      , m_file(m_filename)
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

        if ( m_row[1] == "allocate" ) {
          replay_map_insert();
        }
        else if ( m_row[1] == "deallocate" ) {
          replay_map_remove();
        }
        else {
          std::cout << m_row[1] << "\n";
        }
      }
    }

  private:
    std::string m_filename;
    std::ifstream m_file;
    umpire::util::AllocationMap m_allocation_map;
    std::unordered_map<void*, umpire::util::AllocationRecord> m_mapped_ptrs;    // key(alloc_ptr), val(amap)
    CSVRow m_row;

    template <typename T>
    void get_from_string( const std::string& s, T& val )
    {
        std::istringstream ss(s);
        ss >> val;
    }

    void replay_allocation_map_valid( char* alloc, std::size_t size )
    {
      for ( std::size_t i = 0; i < size; i += 2 ) {
        auto rec = m_allocation_map.find((void*)(&alloc[i]));
        if ( rec == nullptr ) {
          std::cerr << "No AllocationRecord Found: "
            << " [" << (void*)alloc << " - " << (void*)(alloc+size) << "] "
            << "addr(" << (void*)(&alloc[i]) << ") "
            << "index(" << i << ") "
            << "size(" << size << ")\n";
          m_allocation_map.printAll();
        }
      }
    }

    void replay_map_insert( void )
    {
      std::size_t alloc_size;
      void* alloc_ptr;

      get_from_string(m_row[m_row.size() - 1], alloc_ptr);
      get_from_string(m_row[m_row.size() - 3], alloc_size);

      umpire::util::AllocationRecord amap{alloc_ptr, alloc_size, nullptr};
      m_allocation_map.insert(alloc_ptr, amap);

      m_mapped_ptrs[alloc_ptr] = amap;

      replay_allocation_map_valid((char*)alloc_ptr, alloc_size);
    }

    void replay_map_remove( void )
    {
      void* alloc_ptr;

      get_from_string(m_row[m_row.size() - 2], alloc_ptr);

      auto iter = m_mapped_ptrs.find(alloc_ptr);
      if ( iter == m_mapped_ptrs.end() ) {
        std::cerr << "Duplicate attempt to remove " << alloc_ptr << " from map (ignored)" <<  std::endl;
        return;
      }

      auto saved_arec = iter->second;
      auto arec = m_allocation_map.remove(alloc_ptr);

      // TODO Make sure this change is ok with @Marty
      if ( arec.m_ptr != saved_arec.m_ptr )
        std::cerr << "Warning: allocation map mismatch\n";

      m_mapped_ptrs.erase(alloc_ptr);
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

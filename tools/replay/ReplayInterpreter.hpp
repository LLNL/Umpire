//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef REPLAY_ReplayInterpreter_HPP
#define REPLAY_ReplayInterpreter_HPP

#include <fstream>
#include <sstream>
#include <string>

#include "ReplayOperationManager.hpp"
#include "umpire/tpl/json/json.hpp"

class ReplayInterpreter {
  public:
    void buildOperations();
    void runOperations(bool gather_statistics);
    void printInfo();
    bool compareOperations(ReplayInterpreter& rh);

    ReplayInterpreter( std::string in_file_name );
    ~ReplayInterpreter();

    ReplayFile* m_ops{nullptr};

  private:
    using AllocatorIndex = int;
    using AllocatorFromLog = uint64_t;
    using AllocationFromLog = uint64_t;
    using AllocatorIndexMap = std::unordered_map<AllocatorFromLog, AllocatorIndex>;
    using AllocationAllocatorMap = std::unordered_map<AllocationFromLog, AllocatorIndex>;

    std::string m_input_file_name;
    std::ifstream m_input_file;
    std::unordered_map<std::string, void*> m_allocated_ptrs;    // key(alloc_ptr), val(replay_alloc_ptr)
    std::unordered_map<std::string, AllocatorIndex> m_allocator_index;
    std::string m_line;
    nlohmann::json m_json;
    std::vector<std::string> m_row;
    AllocatorIndexMap m_allocator_indices;
    AllocationAllocatorMap m_allocation_id;

    int m_log_version_major;
    int m_log_version_minor;
    int m_log_version_patch;

    template <typename T> void get_from_string( const std::string& s, T& val );

    void strip_off_base(std::string& s);
    void replay_compileMemoryResource( void );
    void replay_compileAllocator( void );
    void replay_compileAllocate( void );
    void replay_compileDeallocate( void );
    void replay_compileCoalesce( void );
    void replay_compileRelease( void );
    void printAllocators(ReplayFile* optable);
};

#include "ReplayInterpreter.inl"

#endif // REPLAY_ReplayInterpreter_HPP

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

#include "util/ReplayOperationManager.hpp"
#include "umpire/tpl/json/json.hpp"

class ReplayInterpreter {
  public:
    void buildOperations(void);
    void buildAllocMapOperations(void);
    void runOperations(bool gather_statistics);

    //
    // Return: > 0 success, 0 eof, < 0 error
    //
    int getSymbolicOperation( std::string& raw_line, std::string& sym_line );

    ReplayInterpreter( std::string in_file_name );

  private:
    using AllocatorIndex = int;
    using AllocatorFromLog = uint64_t;
    using AllocationFromLog = uint64_t;
    using AllocatorIndexMap = std::unordered_map<AllocatorFromLog, AllocatorIndex>;
    using AllocationAllocatorMap = std::unordered_map<AllocationFromLog, AllocatorIndex>;

    std::ifstream m_input_file;
    std::unordered_map<std::string, void*> m_allocated_ptrs;    // key(alloc_ptr), val(replay_alloc_ptr)
    std::string m_line;
    nlohmann::json m_json;
    std::vector<std::string> m_row;
    AllocatorIndex m_num_allocators;
    AllocatorIndexMap m_allocator_indices;
    AllocationAllocatorMap m_allocation_id;
    ReplayOperationManager m_operation_mgr;
    uint64_t m_op_seq;
    std::stringstream compare_ss;

    template <typename T> void get_from_string( const std::string& s, T& val );

    void strip_off_base(std::string& s);
    void replay_makeAllocator( void );
    void replay_makeMemoryResource( void );
    void replay_allocate( void );
    void replay_deallocate( void );
    void replay_coalesce( void );
    void replay_release( void );
    void replay_makeAllocationMapInsert( void );
    void replay_makeAllocationMapFind( void );
    void replay_makeAllocationMapRemove( void );
    void replay_makeAllocationMapClear( void );
};

#include "util/ReplayInterpreter.inl"

#endif // REPLAY_ReplayInterpreter_HPP

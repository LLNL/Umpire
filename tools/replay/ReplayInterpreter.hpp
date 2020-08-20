//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef REPLAY_ReplayInterpreter_HPP
#define REPLAY_ReplayInterpreter_HPP

#if !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "ReplayOperationManager.hpp"
#include "ReplayOptions.hpp"
#include "umpire/tpl/json/json.hpp"

class ReplayInterpreter {
  public:
    void buildOperations();
    void runOperations();
    void printInfo();
    bool compareOperations(ReplayInterpreter& rh);

    ReplayInterpreter( const ReplayOptions& options );
    ~ReplayInterpreter();

    ReplayFile* m_ops{nullptr};

  private:
    using AllocatorIndex = int;
    using AllocatorFromLog = uint64_t;
    using AllocationFromLog = uint64_t;
    using AllocatorIndexMap = std::unordered_map<AllocatorFromLog,
                                                  AllocatorIndex>;
    using AllocationAllocatorMap = std::unordered_map<AllocationFromLog,
                                                  AllocatorIndex>;

    ReplayOptions m_options;
    std::ifstream m_input_file;
    std::unordered_map<std::string, void*> m_allocated_ptrs;
    std::unordered_map<std::string, AllocatorIndex> m_allocator_index;
    std::string m_line;
    nlohmann::json m_json;
    std::vector<std::string> m_row;
    AllocatorIndexMap m_allocator_indices;
    AllocationAllocatorMap m_allocation_id;
    std::unordered_set<AllocationFromLog> m_external_registrations;
    bool m_replaying_reallocate{false};
    std::size_t m_line_number{0};
    bool m_allocation_in_process{false};

    int m_log_version_major;
    int m_log_version_minor;
    int m_log_version_patch;

    template <typename T> void get_from_string( const std::string& s, T& val );

    void strip_off_base(std::string& s);
    void replay_compileMemoryResource( void );
    void replay_compileSetDefaultAllocator( void );
    void replay_processMapInsert( void );
    void replay_processMapRemove( void );
    void replay_compileAllocator( void );
    void replay_compileReallocate( void );
    void replay_compileReallocate_ex( void );
    void replay_compileAllocate( void );
    bool replay_compileDeallocate( void );
    void replay_compileCoalesce( void );
    void replay_compileRelease( void );
    void replay_compileCopy( void );
    int getAllocatorIndex(std::string ref_s);
    uint64_t getPointer(std::string ptr_name);
    void printAllocators(ReplayFile* optable);
};

#include "ReplayInterpreter.inl"

#endif //!defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
#endif // REPLAY_ReplayInterpreter_HPP

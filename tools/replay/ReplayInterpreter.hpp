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
    bool m_make_allocation_in_progress{false};
    bool m_make_allocator_in_progress{false};

    int m_log_version_major;
    int m_log_version_minor;
    int m_log_version_patch;
    std::size_t m_mpi_ops{0};
    std::size_t m_allocation_map_insert_ops{0};
    std::size_t m_allocation_map_insert_due_to_make_allocator{0};
    std::size_t m_allocation_map_insert_due_to_allocation{0};
    std::size_t m_allocation_map_insert_due_to_reallocate{0};
    std::size_t m_allocation_map_insert_rogue_ignored{0};

    std::size_t m_allocation_map_remove_ops{0};
    std::size_t m_allocation_map_remove_rogue_ignored{0};

    std::size_t m_allocation_map_find_ops{0};
    std::size_t m_allocation_map_clear_ops{0};
    std::size_t m_make_allocator_ops{0};
    std::size_t m_make_memory_resource_ops{0};
    std::size_t m_copy_ops{0};
    std::size_t m_memset_ops{0};
    std::size_t m_move_ops{0};
    std::size_t m_reallocate_ex_ops{0};
    std::size_t m_reallocate_ops{0};
    std::size_t m_set_default_allocator_ops{0};
    std::size_t m_allocate_ops{0};
    std::size_t m_deallocate_ops{0};
    std::size_t m_deallocate_due_to_reallocate{0};
    std::size_t m_deallocate_rogue_ignored{0};

    std::size_t m_coalesce_ops{0};
    std::size_t m_release_ops{0};
    std::size_t m_version_ops{0};

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
    void replay_compileMove( void );
    void replay_compileMemset( void );
    int getAllocatorIndex(std::string ref_s);
    uint64_t getPointer(std::string ptr_name);
    void printAllocators(ReplayFile* optable);
};

#include "ReplayInterpreter.inl"

#endif //!defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
#endif // REPLAY_ReplayInterpreter_HPP

//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef REPLAY_ReplayInterpreter_HPP
#define REPLAY_ReplayInterpreter_HPP

#if !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "ReplayOperationManager.hpp"
#include "ReplayOptions.hpp"
#include "umpire/event/event.hpp"
#include "umpire/event/json_file_store.hpp"
#include "umpire/json/json.hpp"

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
    using AllocatorIndexMap = std::unordered_map<AllocatorFromLog, AllocatorIndex>;
    using AllocationAllocatorMap = std::unordered_map<std::string, AllocatorIndex>;

    ReplayOptions m_options;
    std::unordered_map<std::string, AllocatorIndex> m_allocator_index;
    std::string m_line;
    umpire::event::event m_event;
    std::vector<std::string> m_row;
    AllocatorIndexMap m_allocator_indices;
    AllocationAllocatorMap m_allocation_id;
    bool m_replaying_reallocate{false};
    std::size_t m_line_number{0};

    int m_log_version_major;
    int m_log_version_minor;
    int m_log_version_patch;
    std::size_t m_register_external_pointer{0};
    std::size_t m_deregister_external_pointer{0};

    std::size_t m_make_allocator_ops{0};
    std::size_t m_make_memory_resource_ops{0};
    std::size_t m_copy_ops{0};
    std::size_t m_move_ops{0};
    std::size_t m_reallocate_ops{0};
    std::size_t m_set_default_allocator_ops{0};
    std::size_t m_allocate_ops{0};
    std::size_t m_deallocate_ops{0};
    std::size_t m_deallocate_rogue_ignored{0};

    std::size_t m_coalesce_ops{0};
    std::size_t m_release_ops{0};
    std::size_t m_version_ops{0};

    template <typename T> void get_from_string( const std::string& s, T& val );

    void strip_off_base(std::string& s);
    void compile_make_memory_resource();
    void compile_set_default_allocator();
    void compile_make_allocator();
    void compile_reallocate();
    void compile_allocate();
    bool compile_deallocate();
    void compile_coalesce();
    void compile_release();
    int getAllocatorIndex(const std::string& ref_s);
    uint64_t getPointer(std::string ptr_name);
    void printAllocators(ReplayFile* optable);
    std::string printAllocatorInfo(ReplayFile::AllocatorTableEntry* allocator);
};

#include "ReplayInterpreter.inl"

#endif //!defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
#endif // REPLAY_ReplayInterpreter_HPP

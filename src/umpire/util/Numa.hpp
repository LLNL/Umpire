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
#ifndef UMPIRE_Numa_HPP
#define UMPIRE_Numa_HPP

#include <cstddef>
#include <vector>

namespace umpire {
namespace numa {

// The node type - host nodes have associated cpus.
enum class NodeType : int { Host, Device };

void* allocate_on_node(const std::size_t bytes, const std::size_t node);

void deallocate(void* ptr);

void* reallocate(void* ptr, const std::size_t new_bytes);

std::size_t node_count();

std::vector<std::size_t> get_host_nodes();

std::vector<std::size_t> get_device_nodes();

std::size_t preferred_node();

NodeType node_type(const std::size_t node);

} // end of namespace numa
} // end of namespace umpire

#endif // UMPIRE_Numa_HPP

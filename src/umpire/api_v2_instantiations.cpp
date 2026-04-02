//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "../../include/umpire/allocator.hpp"
#include "umpire/resource/host_memory.hpp"
#include "umpire/strategy/fixed_pool.hpp"
#include "umpire/strategy/thread_safe.hpp"

namespace umpire {

namespace resource {

template class host_memory<malloc_allocator, true>;
template class host_memory<malloc_allocator, false>;

} // namespace resource

template class umpire::allocator<char, resource::host_memory<>>;
template class umpire::allocator<int, resource::host_memory<>>;
template class umpire::allocator<double, resource::host_memory<>>;
template class strategy::fixed_pool<resource::host_memory<>>;
template class strategy::thread_safe<resource::host_memory<>>;

} // namespace umpire

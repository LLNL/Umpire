//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/resource/host_memory.hpp"
#include "umpire/strategy/NamedAllocationStrategy.hpp"

namespace {

using host_memory = umpire::resource::host_memory<>;

void mark_phase(const char* phase)
{
  umpire::mark_event(phase);
}

} // namespace

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto host_allocator = rm.getAllocator("HOST");
  auto& host = host_memory::get();

  mark_phase("v1_control_begin");
  void* v1 = host_allocator.allocate(32);
  host_allocator.deallocate(v1);
  mark_phase("v1_control_end");

  mark_phase("v2_direct_begin");
  void* v2_direct = host.allocate(32);
  host.deallocate(v2_direct);
  mark_phase("v2_direct_end");

  mark_phase("v2_rm_deallocate_begin");
  void* v2_rm_deallocate = host.allocate(32);
  rm.deallocate(v2_rm_deallocate);
  mark_phase("v2_rm_deallocate_end");

  mark_phase("v2_rm_copy_begin");
  auto* v2_src = static_cast<unsigned char*>(host.allocate(48));
  auto* v1_dst = static_cast<unsigned char*>(host_allocator.allocate(48));
  rm.copy(v1_dst + 8, v2_src + 4, 16);
  host.deallocate(v2_src);
  host_allocator.deallocate(v1_dst);
  mark_phase("v2_rm_copy_end");

  mark_phase("v2_rm_reallocate_begin");
  void* v2_rm_reallocate = host.allocate(32);
  void* resized = rm.reallocate(v2_rm_reallocate, 64, host_allocator);
  rm.deallocate(resized);
  mark_phase("v2_rm_reallocate_end");

  auto named_allocator = rm.makeAllocator<umpire::strategy::NamedAllocationStrategy>("API_V2_REPLAY_MOVE_DEST", host_allocator);

  mark_phase("v2_rm_move_begin");
  void* v2_rm_move = host.allocate(40);
  void* moved = rm.move(v2_rm_move, named_allocator);
  named_allocator.deallocate(moved);
  mark_phase("v2_rm_move_end");

  return 0;
}

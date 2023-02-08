//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef REPLAY_ReplayOperationManager_HPP
#define REPLAY_ReplayOperationManager_HPP

#if !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
#include <iostream>
#include <cstdint>
#include <vector>

#include "ReplayFile.hpp"
#include "ReplayOptions.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationAdvisor.hpp"
#include "umpire/strategy/SizeLimiter.hpp"
#include "umpire/strategy/MixedPool.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/strategy/SlotPool.hpp"
#include "umpire/strategy/ThreadSafeAllocator.hpp"
#include "umpire/util/AllocationRecord.hpp"
#include "umpire/ResourceManager.hpp"

class ReplayOperationManager {
public:
  ReplayOperationManager( const ReplayOptions& options, ReplayFile* rFile,
      ReplayFile::Header* Operations );
  ~ReplayOperationManager();

  void runOperations();

private:
  std::map<std::string, std::vector< std::pair<size_t, std::size_t>>> m_stat_series;
  ReplayOptions m_options;
  ReplayFile* m_replay_file;
  ReplayFile::Header* m_ops_table;

  void makeAllocator(ReplayFile::Operation* op);
  void makeAllocate(ReplayFile::Operation* op);
  void makeDeallocate(ReplayFile::Operation* op);
  void makeSetDefaultAllocator(ReplayFile::Operation* op);
  void makeCopy(ReplayFile::Operation* op);
  void makeReallocate(ReplayFile::Operation* op);
  void makeReallocate_ex(ReplayFile::Operation* op);
  void makeCoalesce(ReplayFile::Operation* op);
  void makeRelease(ReplayFile::Operation* op);
  void dumpStats();
};

#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
#endif // REPLAY_ReplayOperationManager_HPP

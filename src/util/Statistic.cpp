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

#include "umpire/util/Statistic.hpp"

namespace umpire {
namespace util {

Statistic::Statistic(const std::string& name) :
  m_name(name),
  m_allocation_statistics(),
  m_operation_statistics()
{
}

Statistic::~Statistic()
{
}

void
Statistic::recordAllocationStatistic(AllocationStatistic stat)
{
  stat.timestamp = std::chrono::system_clock::now();
  m_allocation_statistics.push_back(stat);
}

void
Statistic::recordOperationStatistic(OperationStatistic&& stat)
{
  m_operation_statistics.push_back(stat);
}

void
Statistic::printData(std::ostream& stream)
{
}


} // end of namespace util
} // end of namespace umpire

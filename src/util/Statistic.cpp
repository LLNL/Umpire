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

#include <iostream>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace util {

Statistic::Statistic(const std::string& name, Statistic::StatisticType type) :
  m_name(name),
  m_type(type),
  m_statistic_times(),
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
  if (m_type == ALLOC_STAT) {
    m_statistic_times.push_back(std::chrono::system_clock::now());
    m_allocation_statistics.push_back(stat);
  } else {
    UMPIRE_ERROR("Cannot record AllocationStatistic in Statistic with type OP_STAT");
  }
}

void
Statistic::recordOperationStatistic(OperationStatistic stat)
{
  if (m_type == OP_STAT) {
    m_statistic_times.push_back(std::chrono::system_clock::now());
    m_operation_statistics.push_back(stat);
  } else {
    UMPIRE_ERROR("Cannot record OperationStatistic in Statistic with type ALLOC_STAT");
  }
}

void
Statistic::printData(std::ostream& stream)
{
  if (m_type == ALLOC_STAT) {
    std::cout << "Allocation statistic: " << m_name << std::endl;
    for (int i = 0; i < m_allocation_statistics.size(); i++) {
      std::cout << m_statistic_times[i].time_since_epoch().count() << ", ";
      auto record = m_allocation_statistics[i];
      std::cout << "{ " << record.ptr << ", " << record.size << ", " << record.event << "}" << std::endl;
    }
  } else if (m_type == OP_STAT) {
    std::cout << "Operation statistic: " << m_name << std::endl;
    for (int i = 0; i < m_operation_statistics.size(); i++) {
      std::cout << m_statistic_times[i].time_since_epoch().count() << ", ";
      auto record = m_operation_statistics[i];
      std::cout << "{ " << record.src_ptr << ", " << record.dst_ptr << ", " << record.size << ", " << record.event << "}" << std::endl;
    }
  }
}

Statistic::StatisticType
Statistic::getType()
{
  return m_type;
}


} // end of namespace util
} // end of namespace umpire

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
#ifndef UMPIRE_Statistic_HPP
#define UMPIRE_Statistic_HPP

#include <chrono>
#include <vector>
#include <string>

namespace umpire {
namespace util {

class StatisticsDatabase;

class Statistic {
  friend class StatisticsDatabase;
  public:
    ~Statistic();

    struct AllocationStatistic {
      void* ptr;
      size_t size;
      std::string event;
      std::chrono::time_point<std::chrono::system_clock> timestamp;
    }; 

    struct OperationStatistic {
      void* src_ptr;
      void* dst_ptr;
      size_t size;
      std::string event;
      std::chrono::time_point<std::chrono::system_clock> timestamp;
    };

    void recordAllocationStatistic(AllocationStatistic stat);
    void recordOperationStatistic(OperationStatistic&& stat);

    void printData(std::ostream& stream);

  protected:
    Statistic(const std::string& name);

  private:
    std::string m_name;
    std::vector<AllocationStatistic> m_allocation_statistics;
    std::vector<OperationStatistic> m_operation_statistics;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_Statistic_HPP

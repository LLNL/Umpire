//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_StatisticsDatabase_HPP
#define UMPIRE_StatisticsDatabase_HPP

#include <map>
#include <memory>
#include <ostream>

#include "umpire/util/Statistic.hpp"

namespace umpire {
namespace util {

class StatisticsDatabase {
 public:
  static StatisticsDatabase* getDatabase();

  std::shared_ptr<Statistic> getStatistic(const std::string& name);

  void printStatistics(std::ostream& stream) noexcept;

 private:
  StatisticsDatabase() noexcept;

  StatisticsDatabase(const StatisticsDatabase&) = delete;
  StatisticsDatabase& operator=(const StatisticsDatabase&) = delete;

  static StatisticsDatabase* s_statistics_database_instance;

  std::map<std::string, std::shared_ptr<Statistic>> m_statistics;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_StatisticsDatabase_HPP

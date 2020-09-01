//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/util/StatisticsDatabase.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace util {

StatisticsDatabase* StatisticsDatabase::s_statistics_database_instance(nullptr);

StatisticsDatabase* StatisticsDatabase::getDatabase()
{
  if (!s_statistics_database_instance)
    s_statistics_database_instance = new StatisticsDatabase();

  return s_statistics_database_instance;
}

std::shared_ptr<Statistic> StatisticsDatabase::getStatistic(
    const std::string& name)
{
  auto statistic = m_statistics.find(name);
  std::shared_ptr<Statistic> stat;

  if (statistic == m_statistics.end()) {
    stat = std::shared_ptr<Statistic>(new Statistic(name));
    m_statistics[name] = stat;
  } else {
    stat = statistic->second;
  }

  return stat;
}

StatisticsDatabase::StatisticsDatabase() noexcept : m_statistics()
{
}

void StatisticsDatabase::printStatistics(std::ostream& stream) noexcept
{
  stream << "umpire::util::StatisticsDatabase contains " << m_statistics.size()
         << " statistics" << std::endl;
  for (auto& stat : m_statistics) {
    stat.second->printData(stream);
  }
}

} // end of namespace util
} // end of namespace umpire

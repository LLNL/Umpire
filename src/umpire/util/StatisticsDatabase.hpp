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
#ifndef UMPIRE_StatisticsDatabase_HPP
#define UMPIRE_StatisticsDatabase_HPP

#include <ostream>
#include <map>
#include <memory>

#include "umpire/util/Statistic.hpp"

namespace umpire {
namespace util {

class StatisticsDatabase {
  public:
    static StatisticsDatabase* getDatabase();

    std::shared_ptr<Statistic> getStatistic(
        const std::string& name);

    void printStatistics(std::ostream& stream) noexcept;
  private:
    StatisticsDatabase() noexcept;

    StatisticsDatabase (const StatisticsDatabase&) = delete;
    StatisticsDatabase& operator= (const StatisticsDatabase&) = delete;

    static StatisticsDatabase* s_statistics_database_instance;

    std::map<std::string, std::shared_ptr<Statistic> > m_statistics;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_StatisticsDatabase_HPP

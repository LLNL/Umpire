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

#include "conduit/conduit.hpp"

namespace umpire {
namespace util {

class StatisticsDatabase;

class Statistic {
  friend class StatisticsDatabase;
  public:
    ~Statistic() noexcept;

    void recordStatistic(conduit::Node&& n);

    void printData(std::ostream& stream) noexcept;

  protected:
    Statistic(const std::string& name) noexcept;

  private:
    std::string m_name;
    size_t m_counter;

    conduit::Node m_data;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_Statistic_HPP

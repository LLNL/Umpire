//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Statistic_HPP
#define UMPIRE_Statistic_HPP

#include <chrono>
#include <string>
#include <vector>

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
  std::size_t m_counter;

  conduit::Node m_data;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_Statistic_HPP

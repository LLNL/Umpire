//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/util/Statistic.hpp"

#include <iostream>

#include "conduit.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace util {

Statistic::Statistic(const std::string& name) noexcept
    : m_name(name), m_counter(), m_data()
{
  m_data["name"] = name;
}

Statistic::~Statistic() noexcept
{
}

void Statistic::recordStatistic(conduit::Node&& stat)
{
  auto time = std::chrono::time_point_cast<std::chrono::nanoseconds>(
                  std::chrono::system_clock::now())
                  .time_since_epoch();
  stat["timestamp"] = static_cast<long>(time.count());

  m_data["statistics"].append().set(stat);
}

void Statistic::printData(std::ostream& stream) noexcept
{
  m_data.print();
}

} // end of namespace util
} // end of namespace umpire

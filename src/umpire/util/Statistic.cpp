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

#include "conduit.hpp"

namespace umpire {
namespace util {

Statistic::Statistic(const std::string& name) noexcept:
  m_name(name),
  m_counter(),
  m_data()
{
  m_data["name"] = name;
}

Statistic::~Statistic() noexcept
{
}

void
Statistic::recordStatistic(conduit::Node&& stat)
{
  auto time = std::chrono::time_point_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now()).time_since_epoch();
  stat["timestamp"] = static_cast<long>(time.count());

  m_data["statistics"].append().set(stat);
}

void
Statistic::printData(std::ostream& stream) noexcept
{
  m_data.print();
}

} // end of namespace util
} // end of namespace umpire

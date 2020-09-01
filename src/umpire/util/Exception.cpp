//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/util/Exception.hpp"

#include <sstream>

namespace umpire {
namespace util {

Exception::Exception(const std::string& message, const std::string& file,
                     int line)
    : m_message(message), m_file(file), m_line(line)
{
  m_what = this->message();
}

std::string Exception::message() const
{
  std::stringstream oss;
  oss << "! Umpire Exception [" << m_file << ":" << m_line << "]: ";
  oss << m_message;
  return oss.str();
}

const char* Exception::what() const throw()
{
  return m_what.c_str();
}

} // end of namespace util
} // end of namespace umpire

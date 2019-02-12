//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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
#include "umpire/util/Exception.hpp"

#include <sstream>

namespace umpire {
namespace util {

Exception::Exception(
    const std::string& message,
    const std::string &file,
    int line) :
  m_message(message),
  m_file(file),
  m_line(line)
{
  m_what = this->message();
}

std::string
Exception::message() const
{
  std::stringstream oss;
  oss << "! Umpire Exception [" << m_file << ":" << m_line << "]: ";
  oss << m_message;
  return oss.str();
}

const char*
Exception::what() const throw()
{
  return m_what.c_str();
}

} // end of namespace util
} // end of namespace umpire

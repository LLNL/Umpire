//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Exception_HPP
#define UMPIRE_Exception_HPP

#include <exception>
#include <string>

namespace umpire {
namespace util {

class Exception : public std::exception {
 public:
  Exception(const std::string& msg, const std::string& file, int line);

  virtual ~Exception() = default;

  std::string message() const;
  virtual const char* what() const throw();

 private:
  std::string m_message;
  std::string m_file;
  int m_line;

  std::string m_what;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_Exception_HPP

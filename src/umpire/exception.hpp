//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <exception>

namespace umpire {

class exception : public std::exception {
  public:
    exception(const std::string& msg,
        const std::string &file,
        int line);

    virtual ~exception() = default;

    std::string message() const;
    virtual const char* what() const throw();

  private:
    std::string m_message;
    std::string m_file;
    int m_line;

    std::string m_what;
};

} // end of namespace umpire

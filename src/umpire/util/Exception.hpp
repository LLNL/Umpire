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
#ifndef UMPIRE_Exception_HPP
#define UMPIRE_Exception_HPP

#include <string>
#include <exception>

namespace umpire {
namespace util {

class Exception : public std::exception {
  public:
    Exception(std::string  msg,
        std::string file,
        int line);

    ~Exception() override = default;

    std::string message() const;
    const char* what() const throw() override;

  private:
    std::string m_message;
    std::string m_file;
    int m_line;

    std::string m_what;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_Exception_HPP

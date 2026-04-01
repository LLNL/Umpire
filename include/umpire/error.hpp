//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_error_HPP
#define UMPIRE_error_HPP

#include <cstddef>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>

#include "umpire/util/backtrace.hpp"

namespace umpire {

class Allocator;

namespace detail {

inline std::string format_exception_message(const char* type,
                                            const std::string& msg,
                                            const std::string& file,
                                            int line)
{
  if (file.empty()) {
    return msg;
  }

  umpire::util::backtrace bt;
  umpire::util::backtracer<umpire::util::trace_always>::get_backtrace(bt);

  std::stringstream oss;
  oss << "! Umpire " << type << " [" << file << ":" << line << "]: ";
  oss << msg;
  oss << std::endl << umpire::util::backtracer<umpire::util::trace_always>::print(bt) << std::endl;

  return oss.str();
}

} // namespace detail

class runtime_error : public std::runtime_error {
public:
  explicit runtime_error(const std::string& msg)
    : std::runtime_error(msg)
    , m_message(msg)
    , m_what(msg)
  {
  }

  runtime_error(const std::string& msg, const std::string& file, int line)
    : std::runtime_error(msg)
    , m_message(msg)
    , m_file(file)
    , m_line(line)
    , m_what(detail::format_exception_message("runtime_error", msg, file, line))
  {
  }

  virtual ~runtime_error() = default;

  std::string message() const
  {
    return detail::format_exception_message("runtime_error", m_message, m_file, m_line);
  }

  const char* what() const noexcept override
  {
    return m_what.c_str();
  }

private:
  std::string m_message;
  std::string m_file;
  int m_line{0};
  std::string m_what;
};

class logic_error : public std::logic_error {
public:
  explicit logic_error(const std::string& msg)
    : std::logic_error(msg)
    , m_message(msg)
    , m_what(msg)
  {
  }

  logic_error(const std::string& msg, const std::string& file, int line)
    : std::logic_error(msg)
    , m_message(msg)
    , m_file(file)
    , m_line(line)
    , m_what(detail::format_exception_message("logic_error", msg, file, line))
  {
  }

  virtual ~logic_error() = default;

  std::string message() const
  {
    return detail::format_exception_message("logic_error", m_message, m_file, m_line);
  }

  const char* what() const noexcept override
  {
    return m_what.c_str();
  }

private:
  std::string m_message;
  std::string m_file;
  int m_line{0};
  std::string m_what;
};

class out_of_memory : public std::bad_alloc {
public:
  explicit out_of_memory(const std::string& msg)
    : m_message(msg)
    , m_what(msg)
  {
  }

  out_of_memory(const std::string& msg, const std::string& file, int line)
    : m_message(msg)
    , m_file(file)
    , m_line(line)
    , m_what(detail::format_exception_message("out_of_memory", msg, file, line))
  {
  }

  virtual ~out_of_memory() = default;

  std::string message() const
  {
    return detail::format_exception_message("out_of_memory", m_message, m_file, m_line);
  }

  const char* what() const noexcept override
  {
    return m_what.c_str();
  }

  std::size_t requested_size()
  {
    return m_requested;
  }

  int get_allocator_id()
  {
    return m_allocator;
  }

private:
  void set_allocator_id(int id)
  {
    m_allocator = id;
  }

  void set_requested_size(std::size_t s)
  {
    m_requested = s;
  }

  std::string m_message;
  std::string m_file;
  int m_line{0};
  std::string m_what;
  std::size_t m_requested{0};
  int m_allocator{-1};

  friend class Allocator;
};

class unknown_allocation : public std::runtime_error {
public:
  explicit unknown_allocation(const std::string& msg)
    : std::runtime_error(msg)
    , m_message(msg)
    , m_what(msg)
  {
  }

  unknown_allocation(const std::string& msg, const std::string& file, int line)
    : std::runtime_error(msg)
    , m_message(msg)
    , m_file(file)
    , m_line(line)
    , m_what(detail::format_exception_message("unknown_allocation", msg, file, line))
  {
  }

  virtual ~unknown_allocation() = default;

  std::string message() const
  {
    return detail::format_exception_message("unknown_allocation", m_message, m_file, m_line);
  }

  const char* what() const noexcept override
  {
    return m_what.c_str();
  }

  void* get_pointer() const
  {
    return m_pointer;
  }

private:
  void set_pointer(void* p)
  {
    m_pointer = p;
  }

  std::string m_message;
  std::string m_file;
  int m_line{0};
  std::string m_what;
  void* m_pointer{nullptr};

  friend class Allocator;
};

} // namespace umpire

#endif // UMPIRE_error_HPP

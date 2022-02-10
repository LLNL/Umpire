//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_event_HPP
#define UMPIRE_event_HPP

#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include "camp/camp.hpp"
#include "umpire/event/recorder_factory.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace event {

enum class category { operation, statistic, metadata };

class event {
 public:
  class builder;

  std::string name{"anon"};
  ;
  category cat{category::statistic};
  std::map<std::string, std::string> string_args{};
  std::map<std::string, std::uintmax_t> numeric_args{};
  std::map<std::string, std::string> tags{};
  std::chrono::time_point<std::chrono::system_clock> timestamp{std::chrono::system_clock::now()};
};

class event::builder {
 public:
  builder& name(const char* n)
  {
    std::string nm{n};
    e.name = nm;
    return *this;
  }

  builder& name(const std::string& n)
  {
    e.name = n;
    return *this;
  }

  builder& category(category c)
  {
    e.cat = c;
    return *this;
  }

  builder& arg(const std::string& k, void* p)
  {
    std::stringstream ss;
    ss << p;
    std::string pointer{ss.str()};
    e.string_args[k] = pointer;
    return *this;
  }

  builder& arg(const std::string& k, const std::string& v)
  {
    e.string_args[k] = v;
    return *this;
  }

  builder& arg(const std::string& k, const char* v)
  {
    return arg(k, std::string{v});
  }

  builder& arg(const std::string& k, char* v)
  {
    return arg(k, std::string{v});
  }

  builder& arg(const char* k, void* p)
  {
    return arg(std::string{k}, p);
  }

  builder& arg(const char* k, const char* v)
  {
    return arg(std::string{k}, std::string{v});
  }

  builder& arg(const char* k, const std::string& v)
  {
    return arg(std::string{k}, v);
  }

  template <typename T>
  std::enable_if_t<std::is_arithmetic<T>::value, builder&> arg(const std::string& k, T v)
  {
    e.numeric_args[k] = static_cast<std::uintmax_t>(v);
    return *this;
  }

  template <typename T>
  std::enable_if_t<std::is_arithmetic<T>::value, builder&> arg(const char* k, T v)
  {
    return arg(std::string{k}, v);
  }

  template <typename T>
  std::enable_if_t<!std::is_arithmetic<T>::value, builder&> arg(const std::string& k, T v)
  {
    using std::to_string;
    return arg(k, to_string(v));
  }

  template <typename... Ts, std::size_t... N>
  builder& args_impl(std::index_sequence<N...>, Ts... as)
  {
    UMPIRE_USE_VAR(CAMP_EXPAND(arg("arg" + std::to_string(N), as)));
    return *this;
  }

  template <typename... Ts>
  builder& args(Ts... as)
  {
    return args_impl(std::make_index_sequence<sizeof...(Ts)>(), as...);
  }

  builder& tag(const char* t, const char* v)
  {
    std::string tagstr{t};
    std::string value{v};
    e.tags[tagstr] = value;
    return *this;
  }

  builder& tag(const std::string& t, const std::string& v)
  {
    e.tags[t] = v;
    return *this;
  }

  template <typename Recorder = decltype(recorder_factory::get_recorder())>
  void record(Recorder r = recorder_factory::get_recorder())
  {
    r.record(e);
  }

 private:
  event e;
};

namespace {
static const char* replay_env{getenv("UMPIRE_REPLAY")};
static const bool enable_replay{(replay_env != NULL)};
static const char* event_env{getenv("UMPIRE_EVENTS")};
static const bool enable_event{(event_env != NULL)};
static const bool event_build_enabled{enable_replay || enable_event};
} // namespace

template <typename Lambda>
void record(Lambda&& l)
{
  if (event_build_enabled) {
    umpire::event::event::builder e;
    l(e);
    e.record();
  }
}

} // namespace event
} // namespace umpire
#endif

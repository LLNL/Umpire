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

namespace {

bool event_build_enabled()
{
  static char* replay_env = getenv("UMPIRE_REPLAY");
  static bool enable_replay = (replay_env != NULL);

  static char* event_env = getenv("UMPIRE_EVENTS");
  static bool enable_event = (event_env != NULL);

  return (enable_replay || enable_event);
}

} // namespace

enum class category { operation, statistic, metadata };

class event {
 public:
  class builder;

  event()
  {
     if (event_build_enabled()) {
        timestamp = std::chrono::system_clock::now();
     }
  }

  std::string name;
  category cat{category::statistic};
  std::map<std::string, std::string> string_args{};
  std::map<std::string, std::uintmax_t> numeric_args{};
  std::map<std::string, std::string> tags{};
  std::chrono::time_point<std::chrono::system_clock> timestamp{};
};

class event::builder {
 public:
  builder() : event_enabled{event_build_enabled()}
  {
    if (event_enabled) {
      e.name = "anon";
    }
  }

  builder& name(const char* n)
  {
    if (event_enabled) {
      std::string nm{n};
      e.name = nm;
    }
    return *this;
  }

  builder& name(const std::string& n)
  {
    if (event_enabled)
      e.name = n;
    return *this;
  }

  builder& category(category c)
  {
    if (event_enabled)
      e.cat = c;
    return *this;
  }

  builder& arg(const std::string& k, void* p)
  {
    if (event_enabled) {
      std::stringstream ss;
      ss << p;
      std::string pointer{ss.str()};
      e.string_args[k] = pointer;
    }
    return *this;
  }

  builder& arg(const std::string& k, const std::string& v)
  {
    if (event_enabled)
      e.string_args[k] = v;
    return *this;
  }

  builder& arg(const std::string& k, const char* v)
  {
    return event_enabled ? arg(k, std::string{v}) : *this;
  }

  builder& arg(const std::string& k, char* v)
  {
    return event_enabled ? arg(k, std::string{v}) : *this;
  }

  builder& arg(const char* k, void* p)
  {
    return event_enabled ? arg(std::string{k}, p) : *this;
  }

  builder& arg(const char* k, const char* v)
  {
    return event_enabled ? arg(std::string{k}, std::string{v}) : *this;
  }

  builder& arg(const char* k, const std::string& v)
  {
    return event_enabled ? arg(std::string{k}, v) : *this;
  }

  template <typename T>
  std::enable_if_t<std::is_arithmetic<T>::value, builder&> arg(const std::string& k, T v)
  {
    if (event_enabled)
      e.numeric_args[k] = static_cast<std::uintmax_t>(v);
    return *this;
  }

  template <typename T>
  std::enable_if_t<std::is_arithmetic<T>::value, builder&> arg(const char* k, T v)
  {
    return event_enabled ? arg(std::string{k}, v) : *this;
  }

  template <typename T>
  std::enable_if_t<!std::is_arithmetic<T>::value, builder&> arg(const std::string& k, T v)
  {
    if (event_enabled) {
      using std::to_string;
      return arg(k, to_string(v));
    } else {
      return *this;
    }
  }

  template <typename... Ts, std::size_t... N>
  builder& args_impl(std::index_sequence<N...>, Ts... as)
  {
    if (event_enabled) {
      UMPIRE_USE_VAR(CAMP_EXPAND(arg("arg" + std::to_string(N), as)));
    }
    return *this;
  }

  template <typename... Ts>
  builder& args(Ts... as)
  {
    if (event_enabled) {
      return args_impl(std::make_index_sequence<sizeof...(Ts)>(), as...);
    } else {
      return *this;
    }
  }

  builder& tag(const char* t, const char* v)
  {
    if (event_enabled) {
      std::string tagstr{t};
      std::string value{v};
      e.tags[tagstr] = value;
    }
    return *this;
  }

  builder& tag(const std::string& t, const std::string& v)
  {
    if (event_enabled)
      e.tags[t] = v;
    return *this;
  }

  template <typename Recorder = decltype(recorder_factory::get_recorder())>
  void record(Recorder r = recorder_factory::get_recorder())
  {
    if (event_enabled)
      r.record(e);
  }

 private:
  event e;
  bool event_enabled;
};

} // namespace event
} // namespace umpire
#endif

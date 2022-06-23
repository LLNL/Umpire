//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_event_json_HPP
#define UMPIRE_event_json_HPP

#include "umpire/event/event.hpp"
#include "umpire/json/json.hpp"

namespace umpire {
namespace event {

NLOHMANN_JSON_SERIALIZE_ENUM(category, {
                                           {category::operation, "operation"},
                                           {category::statistic, "statistic"},
                                           {category::metadata, "metadata"},
                                       })

inline void to_json(nlohmann::json& j, const event& e)
{
  j = nlohmann::json{
      {"name", e.name},
      {"category", e.cat},
      {"numeric_args", e.numeric_args},
      {"string_args", e.string_args},
      {"tags", e.tags},
      {"timestamp",
       static_cast<long>(
           std::chrono::time_point_cast<std::chrono::nanoseconds>(e.timestamp).time_since_epoch().count())}};
}

inline void from_json(const nlohmann::json& j, event& e)
{
  j.at("name").get_to(e.name);
  j.at("category").get_to(e.cat);
  if (j.find("string_args") != j.end()) {
    j.at("string_args").get_to(e.string_args);
  }
  if (j.find("numeric_args") != j.end()) {
    j.at("numeric_args").get_to(e.numeric_args);
  }
  j.at("tags").get_to(e.tags);

  long dur_ns;
  j.at("timestamp").get_to(dur_ns);
  std::chrono::nanoseconds dur(dur_ns);
  std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> dt_nano(dur);
  // std::chrono::time_point<std::chrono::system_clock> dt(dur);
  e.timestamp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(dt_nano);
}

} // namespace event
} // namespace umpire

#endif // UMPIRE_event_json_HPP

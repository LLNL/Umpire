#ifndef UMPIRE_record_event_HPP
#define  UMPIRE_record_event_HPP

#include "umpire/tpl/influxdb/influxdb.hpp"

#include <cstddef>
#include <string>

namespace umpire {
namespace util {

namespace event {
  struct allocate {};
  struct deallocate {};
  struct current_size {};
  struct actual_size {};
  struct hwm {};
}


namespace {
  static influxdb_cpp::server_info si{"127.0.0.1", 8086, "umpire"};
}

inline void record_event(event::allocate, const std::string& allocator_name, std::size_t size, void* ptr)
{
  influxdb_cpp::builder()
      .meas("events")
      .tag("event", "allocate")
      .tag("allocator", allocator_name)
      .field("size", static_cast<long long>(size))
      .field("ptr", reinterpret_cast<long long>(ptr))
      .post_http(si);
}

inline void record_event(event::deallocate, const std::string& allocator_name, void* ptr)
{
  influxdb_cpp::builder()
      .meas("events")
      .tag("event", "deallocate")
      .tag("allocator", allocator_name)
      .field("ptr", reinterpret_cast<long long>(ptr))
      .post_http(si);
}

inline void record_event(event::current_size, const std::string& allocator_name, std::size_t size)
{
  influxdb_cpp::builder()
      .meas("events")
      .tag("event", "current_size")
      .tag("allocator", allocator_name)
      .field("ptr", static_cast<long long>(size))
      .post_http(si);
}

inline void record_event(event::actual_size, const std::string& allocator_name, std::size_t size)
{
  influxdb_cpp::builder()
      .meas("events")
      .tag("event", "actual_size")
      .tag("allocator", allocator_name)
      .field("ptr", static_cast<long long>(size))
      .post_http(si);
}

inline void record_event(event::hwm, const std::string& allocator_name, std::size_t size)
{
  influxdb_cpp::builder()
      .meas("events")
      .tag("event", "hwm")
      .tag("allocator", allocator_name)
      .field("ptr", static_cast<long long>(size))
      .post_http(si);
}

}
}

#endif  // UMPIRE_record_event_HPP

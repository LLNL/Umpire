#pragma once

namespace umpire {
namespace tracking {

struct trackable
{
  virtual std::size_t get_current_size() const noexcept = 0;
  virtual std::size_t get_actual_size() const noexcept = 0;
  virtual std::size_t get_highwatermark() const noexcept = 0;
};

}
}
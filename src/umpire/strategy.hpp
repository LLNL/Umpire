#pragma once

#include "umpire/strategy/allocation_strategy.hpp"
#include "umpire/strategy/mixed_pool.hpp"
#include "umpire/strategy/dynamic_pool_list.hpp"
#include "umpire/strategy/dynamic_pool_map.hpp"
#include "umpire/strategy/fixed_pool.hpp"
#include "umpire/strategy/monotonic_buffer.hpp"
#include "umpire/strategy/named.hpp"
#include "umpire/strategy/quick_pool.hpp"
#include "umpire/strategy/size_limiter.hpp"
#include "umpire/strategy/slot_pool.hpp"
#include "umpire/strategy/thread_safe.hpp"

namespace umpire {
namespace strategy {

template<typename Memory=memory, bool Tracking=true>
using dynamic_pool = dynamic_pool_map<Memory, Tracking>;

} // end of namespace strategy
} // end of namespace umpire
//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/event/recorder_chain.hpp"

#include "umpire/event/event.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace event {

void recorder_chain::add_store(event_store* store)
{
  if (store != nullptr) {
    m_stores.push_back(store);
  }
}

void recorder_chain::insert(const event& e)
{
  for (auto* store : m_stores) {
    try {
      store->insert(e);
    } catch (const std::exception& ex) {
      UMPIRE_LOG(Warning, "recorder_chain: Failed to record event to store: " << ex.what());
    }
  }
}

void recorder_chain::insert(const allocate& e)
{
  for (auto* store : m_stores) {
    try {
      store->insert(e);
    } catch (const std::exception& ex) {
      UMPIRE_LOG(Warning, "recorder_chain: Failed to record allocate event to store: " << ex.what());
    }
  }
}

void recorder_chain::insert(const named_allocate& e)
{
  for (auto* store : m_stores) {
    try {
      store->insert(e);
    } catch (const std::exception& ex) {
      UMPIRE_LOG(Warning, "recorder_chain: Failed to record named_allocate event to store: " << ex.what());
    }
  }
}

void recorder_chain::insert(const allocate_resource& e)
{
  for (auto* store : m_stores) {
    try {
      store->insert(e);
    } catch (const std::exception& ex) {
      UMPIRE_LOG(Warning, "recorder_chain: Failed to record allocate_resource event to store: " << ex.what());
    }
  }
}

void recorder_chain::insert(const deallocate& e)
{
  for (auto* store : m_stores) {
    try {
      store->insert(e);
    } catch (const std::exception& ex) {
      UMPIRE_LOG(Warning, "recorder_chain: Failed to record deallocate event to store: " << ex.what());
    }
  }
}

void recorder_chain::insert(const deallocate_resource& e)
{
  for (auto* store : m_stores) {
    try {
      store->insert(e);
    } catch (const std::exception& ex) {
      UMPIRE_LOG(Warning, "recorder_chain: Failed to record deallocate_resource event to store: " << ex.what());
    }
  }
}

std::vector<event> recorder_chain::get_events()
{
  // Return events from the first store that supports it
  for (auto* store : m_stores) {
    try {
      auto events = store->get_events();
      if (!events.empty()) {
        return events;
      }
    } catch (const std::exception&) {
      // Try next store
    }
  }
  return {};
}

} // namespace event
} // namespace umpire

//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/event/prometheus_backend.hpp"

#include "umpire/event/event.hpp"
#include "umpire/util/Macros.hpp"

#include <curl/curl.h>

#include <thread>

namespace umpire {
namespace event {

prometheus_backend::prometheus_backend(const std::string& endpoint, int rank, const std::string& job_name,
                                       const std::string& environment, std::chrono::seconds flush_interval)
    : m_remote_write_endpoint(endpoint),
      m_rank(rank),
      m_job_name(job_name),
      m_environment(environment),
      m_flush_interval(flush_interval)
{
  // Initialize libcurl globally (must be called once per process)
  curl_global_init(CURL_GLOBAL_DEFAULT);

  // Start background flush thread
  m_flush_thread = std::thread(&prometheus_backend::flush_loop, this);

  UMPIRE_LOG(Info, "prometheus_backend: Started with endpoint=" << endpoint << " rank=" << rank
                                                                 << " flush_interval=" << flush_interval.count()
                                                                 << "s");
}

prometheus_backend::~prometheus_backend()
{
  // Signal thread to stop
  m_running.store(false, std::memory_order_release);

  // Wait for thread to finish
  if (m_flush_thread.joinable()) {
    m_flush_thread.join();
  }

  // Final flush of remaining metrics
  send_with_retry();

  // Cleanup libcurl
  curl_global_cleanup();

  UMPIRE_LOG(Info, "prometheus_backend: Shutdown complete (send_failures=" << m_send_failures.load() << ")");
}

void prometheus_backend::send_batch(std::span<const event> events)
{
  // Generic events don't have allocation info - skip for now
  (void)events;
}

void prometheus_backend::send_batch(std::span<const allocate> events)
{
  for (const auto& e : events) {
    m_aggregator.record_allocate(e.ptr, e.size, e.ref);
  }
}

void prometheus_backend::send_batch(std::span<const named_allocate> events)
{
  for (const auto& e : events) {
    m_aggregator.record_allocate(e.ptr, e.size, e.ref);
  }
}

void prometheus_backend::send_batch(std::span<const allocate_resource> events)
{
  for (const auto& e : events) {
    m_aggregator.record_allocate(e.ptr, e.size, e.ref);
  }
}

void prometheus_backend::send_batch(std::span<const deallocate> events)
{
  for (const auto& e : events) {
    m_aggregator.record_deallocate(e.ptr, e.ref);
  }
}

void prometheus_backend::send_batch(std::span<const deallocate_resource> events)
{
  for (const auto& e : events) {
    m_aggregator.record_deallocate(e.ptr, e.ref);
  }
}

bool prometheus_backend::is_healthy() const
{
  return m_state.load(std::memory_order_relaxed) != State::FALLBACK;
}

void prometheus_backend::flush_loop()
{
  while (m_running.load(std::memory_order_acquire)) {
    // Sleep for flush interval
    std::this_thread::sleep_for(m_flush_interval);

    // Send metrics to Prometheus
    if (m_state.load(std::memory_order_relaxed) != State::FALLBACK) {
      if (!send_with_retry()) {
        UMPIRE_LOG(Error, "prometheus_backend: All retries failed, entering FALLBACK mode (metrics will not be "
                          "sent to Prometheus)");
        m_state.store(State::FALLBACK, std::memory_order_relaxed);
      }
    }
  }
}

bool prometheus_backend::send_with_retry()
{
  const int max_retries = 3;
  std::chrono::seconds backoff(1);

  for (int attempt = 0; attempt < max_retries; ++attempt) {
    if (attempt > 0) {
      m_state.store(State::RETRYING, std::memory_order_relaxed);
      UMPIRE_LOG(Warning, "prometheus_backend: Retry " << attempt << "/" << max_retries
                                                        << " after " << backoff.count() << "s");
      std::this_thread::sleep_for(backoff);
      backoff *= 2;  // Exponential backoff (1s, 2s, 4s)
    }

    if (send_metrics_to_prometheus()) {
      if (m_state.load(std::memory_order_relaxed) == State::RETRYING) {
        UMPIRE_LOG(Info, "prometheus_backend: Recovered from failure, returning to ACTIVE state");
        m_state.store(State::ACTIVE, std::memory_order_relaxed);
      }
      return true;
    }

    m_send_failures.fetch_add(1, std::memory_order_relaxed);
  }

  return false;
}

// libcurl write callback (required for HTTP POST)
static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp)
{
  (void)contents;
  (void)userp;
  return size * nmemb;  // Discard response body
}

bool prometheus_backend::send_metrics_to_prometheus()
{
  // Render metrics in Prometheus text format
  std::string metrics_text = m_aggregator.render_prometheus_text();

  if (metrics_text.empty()) {
    UMPIRE_LOG(Debug, "prometheus_backend: No metrics to send");
    return true;  // Nothing to send is not an error
  }

  // Initialize libcurl handle
  CURL* curl = curl_easy_init();
  if (!curl) {
    UMPIRE_LOG(Error, "prometheus_backend: Failed to initialize libcurl");
    return false;
  }

  // Set URL
  curl_easy_setopt(curl, CURLOPT_URL, m_remote_write_endpoint.c_str());

  // Set HTTP POST
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, metrics_text.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(metrics_text.size()));

  // Set headers
  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: text/plain; version=0.0.4");
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

  // Set write callback (discard response)
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);

  // Set timeout (10 seconds)
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);

  // Perform HTTP POST
  CURLcode res = curl_easy_perform(curl);

  // Check response code
  long response_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);

  // Cleanup
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  if (res != CURLE_OK) {
    UMPIRE_LOG(Error, "prometheus_backend: HTTP POST failed: " << curl_easy_strerror(res));
    return false;
  }

  if (response_code < 200 || response_code >= 300) {
    UMPIRE_LOG(Error, "prometheus_backend: HTTP POST returned error code " << response_code);
    return false;
  }

  UMPIRE_LOG(Debug, "prometheus_backend: Sent " << metrics_text.size() << " bytes to Prometheus (HTTP "
                                                 << response_code << ")");
  return true;
}

} // namespace event
} // namespace umpire

//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/event/recorder_factory.hpp"

#include "umpire/config.hpp"

#ifdef UMPIRE_ENABLE_SQLITE_EXPERIMENTAL
#include "umpire/event/sqlite_database.hpp"
#else
#include "umpire/event/json_file_store.hpp"
#endif // UMPIRE_ENABLE_SQLITE_EXPERIMENTAL

#include "umpire/event/recorder_chain.hpp"

#ifdef UMPIRE_ENABLE_STREAMING
#include "umpire/event/prometheus_backend.hpp"
#include "umpire/event/streaming_event_sink.hpp"
#endif

#include "umpire/util/Macros.hpp"
#include "umpire/util/io.hpp"

#if !defined(_MSC_VER)
#include <unistd.h> // getpid()
#else
#include <process.h>
#define getpid _getpid
#include <direct.h>
#endif

namespace umpire {
namespace event {

store_type& recorder_factory::get_recorder()
{
  static const std::string filename{
      util::make_unique_filename(util::get_io_output_dir(), util::get_io_output_basename(), getpid(), "stats")};

  // File-based store (always created for replay/debugging)
#ifdef UMPIRE_ENABLE_SQLITE_EXPERIMENTAL
  static sqlite_database db{filename};
#else
  static json_file_store db{filename};
#endif // UMPIRE_ENABLE_SQLITE_EXPERIMENTAL

  // Recorder chain with file store + optional streaming
  static recorder_chain chain;
  static event_store_recorder recorder(&chain);

  // Initialize recorder chain (done once)
  static bool initialized = false;
  if (!initialized) {
    // Add file store to chain
    chain.add_store(&db);

#ifdef UMPIRE_ENABLE_STREAMING
    // Check if streaming is enabled via environment variable
    const char* streaming_backend = std::getenv("UMPIRE_STREAMING_BACKEND");
    if (streaming_backend != nullptr && std::string(streaming_backend) == "prometheus") {
      // Get Prometheus endpoint from environment
      const char* prometheus_endpoint_env = std::getenv("UMPIRE_PROMETHEUS_ENDPOINT");
      std::string prometheus_endpoint =
          prometheus_endpoint_env ? prometheus_endpoint_env : "http://localhost:9090/api/v1/write";

      // Get optional metadata labels
      const char* job_name_env = std::getenv("UMPIRE_JOB_NAME");
      std::string job_name = job_name_env ? job_name_env : "umpire";

      const char* environment_env = std::getenv("UMPIRE_DEPLOYMENT_ENV");
      std::string environment = environment_env ? environment_env : "production";

      // Get flush interval (default: 30 seconds)
      const char* flush_interval_env = std::getenv("UMPIRE_PROMETHEUS_FLUSH_INTERVAL_SEC");
      int flush_interval_sec = flush_interval_env ? std::atoi(flush_interval_env) : 30;

      // Get MPI rank (default: 0 for non-MPI)
      int rank = 0;
#ifdef UMPIRE_ENABLE_MPI
      // If MPI is enabled, get rank from MPI
      int initialized_mpi = 0;
      MPI_Initialized(&initialized_mpi);
      if (initialized_mpi) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      }
#endif

      // Create Prometheus backend and streaming sink
      static prometheus_backend backend(prometheus_endpoint, rank, job_name, environment,
                                        std::chrono::seconds(flush_interval_sec));
      static streaming_event_sink streaming_sink(&backend);

      // Add streaming sink to chain
      chain.add_store(&streaming_sink);

      UMPIRE_LOG(Info, "Streaming enabled: backend=prometheus endpoint=" << prometheus_endpoint << " rank=" << rank
                                                                          << " job=" << job_name);
    }
#endif

    initialized = true;
  }

  static bool info_logged = false;
  if (!info_logged) {
    const char* replay_env = std::getenv("UMPIRE_REPLAY");
    const char* event_env = std::getenv("UMPIRE_EVENTS");
    if (replay_env != nullptr || event_env != nullptr) {
      UMPIRE_LOG(Info,
                 "Event recording enabled via environment variables. "
                 "Note: Variables must be set before program execution.");
    }
    info_logged = true;
  }

  return recorder;
}

} // namespace event
} // namespace umpire

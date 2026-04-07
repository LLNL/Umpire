.. _online_monitoring:

================================
Online Monitoring and Alerting
================================

Umpire provides real-time monitoring and alerting capabilities through integration
with Prometheus and Grafana. This feature allows you to track memory allocation
patterns, detect leaks, and set up alerts in production environments.

Overview
========

The online monitoring system consists of three main components:

1. **Streaming Event Sink**: A thread-safe queue that buffers allocation events
2. **Prometheus Backend**: Aggregates events into metrics and sends them to Prometheus
3. **Grafana Dashboard**: Visualizes metrics and displays alerts

Unlike the file-based replay system (which is optimized for post-mortem forensics),
the streaming system provides **aggregate metrics** for operational monitoring:

- Total allocations/deallocations per rank
- Current memory usage per allocator
- Allocation size distribution histograms
- Leak detection (allocation rate vs deallocation rate)

.. note::
   For detailed per-allocation tracing, use the existing replay file system
   (``UMPIRE_REPLAY=On``). The streaming system is designed for high-level
   monitoring and alerting, not individual allocation queries.

Quick Start
===========

Enabling Streaming
------------------

1. Build Umpire with streaming support enabled:

.. code-block:: bash

   cmake .. -DUMPIRE_ENABLE_STREAMING=On
   make

2. Set environment variables before running your application:

.. code-block:: bash

   export UMPIRE_STREAMING_BACKEND=prometheus
   export UMPIRE_PROMETHEUS_ENDPOINT=http://localhost:9090/api/v1/write
   export UMPIRE_JOB_NAME=my-hpc-job
   export UMPIRE_DEPLOYMENT_ENV=production

3. Run your application as normal. Metrics will be sent to Prometheus every 30 seconds.

Environment Variables
=====================

Core Configuration
------------------

.. list-table::
   :widths: 30 50 20
   :header-rows: 1

   * - Variable
     - Description
     - Default
   * - ``UMPIRE_STREAMING_BACKEND``
     - Backend type (currently only "prometheus" supported)
     - None (disabled)
   * - ``UMPIRE_PROMETHEUS_ENDPOINT``
     - Prometheus remote-write endpoint URL
     - ``http://localhost:9090/api/v1/write``
   * - ``UMPIRE_PROMETHEUS_FLUSH_INTERVAL_SEC``
     - How often to send metrics (seconds)
     - 30

Metadata Labels
---------------

These labels are attached to all metrics for filtering and grouping:

.. list-table::
   :widths: 30 50 20
   :header-rows: 1

   * - Variable
     - Description
     - Default
   * - ``UMPIRE_JOB_NAME``
     - Job name (e.g., "turbulence-sim")
     - "umpire"
   * - ``UMPIRE_DEPLOYMENT_ENV``
     - Environment (e.g., "production", "staging")
     - "production"

Performance Tuning
------------------

.. list-table::
   :widths: 30 50 20
   :header-rows: 1

   * - Variable
     - Description
     - Default
   * - ``UMPIRE_RING_BUFFER_SIZE``
     - Max events buffered before backpressure
     - 16384

.. warning::
   Increasing ``UMPIRE_RING_BUFFER_SIZE`` beyond 32768 may cause memory overhead.
   The default of 16384 is sized for 99th percentile event rates with 2-4 threads.

Metrics Reference
=================

Counters
--------

**umpire_allocations_total**
  Total number of allocation calls (monotonically increasing).

  Labels: ``rank``, ``allocator``, ``job``, ``environment``

  Example query: ``rate(umpire_allocations_total[5m])``

**umpire_deallocations_total**
  Total number of deallocation calls (monotonically increasing).

  Labels: ``rank``, ``allocator``, ``job``, ``environment``

  Example query: ``rate(umpire_deallocations_total[5m])``

Gauges
------

**umpire_bytes_allocated**
  Current bytes allocated (can increase or decrease).

  Labels: ``rank``, ``allocator``, ``job``, ``environment``

  Example query: ``sum by (allocator) (umpire_bytes_allocated)``

Histograms
----------

**umpire_allocation_size_bytes**
  Distribution of allocation sizes in bytes.

  Buckets: ``1KB, 4KB, 16KB, 64KB, 256KB, 1MB, 4MB, 16MB, +Inf``

  Labels: ``rank``, ``job``, ``environment``

  Example queries:

  - Average allocation size: ``rate(umpire_allocation_size_bytes_sum[5m]) / rate(umpire_allocation_size_bytes_count[5m])``
  - 95th percentile: ``histogram_quantile(0.95, rate(umpire_allocation_size_bytes_bucket[5m]))``

Common Queries
==============

Memory Leak Detection
----------------------

Detect if allocation rate exceeds deallocation rate by more than 10%:

.. code-block:: promql

   rate(umpire_allocations_total[5m]) > rate(umpire_deallocations_total[5m]) * 1.1

Total Memory Across All Ranks
------------------------------

.. code-block:: promql

   sum(umpire_bytes_allocated)

Top 5 Allocators by Usage
--------------------------

.. code-block:: promql

   topk(5, sum by (allocator) (umpire_bytes_allocated))

Allocation Rate per Rank
-------------------------

.. code-block:: promql

   rate(umpire_allocations_total[1m])

Setting Up Grafana Dashboard
=============================

Import Pre-Built Dashboard
---------------------------

1. Open Grafana web interface
2. Navigate to **Dashboards** → **Import**
3. Upload ``tools/monitoring/dashboards/umpire-overview.json``
4. Select your Prometheus datasource
5. Click **Import**

The dashboard includes:

- Memory usage over time (per rank and allocator)
- Allocation/deallocation rate charts
- Leak detection visualization
- Top allocators pie chart
- Allocation size distribution heatmap
- Summary statistics (total memory, avg allocation size, active ranks)

Customizing the Dashboard
--------------------------

To modify the dashboard:

1. Click the gear icon (⚙️) in the top right
2. Select **JSON Model**
3. Edit the JSON and save
4. Export the modified dashboard for future use

Setting Up Prometheus Alerts
=============================

Loading Alert Rules
-------------------

1. Copy ``tools/monitoring/alerts/umpire-alerts.yml`` to your Prometheus server
2. Add to ``prometheus.yml``:

.. code-block:: yaml

   rule_files:
     - "umpire-alerts.yml"

3. Restart Prometheus:

.. code-block:: bash

   systemctl restart prometheus

Configuring Alertmanager
-------------------------

To receive notifications (email, Slack, PagerDuty), configure Alertmanager:

.. code-block:: yaml

   # alertmanager.yml
   route:
     group_by: ['alertname', 'rank']
     receiver: 'email-alerts'

   receivers:
     - name: 'email-alerts'
       email_configs:
         - to: 'ops-team@example.com'
           from: 'prometheus@example.com'
           smarthost: 'smtp.example.com:587'

Available Alerts
----------------

The pre-configured alert rules include:

.. list-table::
   :widths: 25 50 25
   :header-rows: 1

   * - Alert Name
     - Trigger Condition
     - Severity
   * - UmpireMemoryLeak
     - Alloc rate > dealloc rate by 20% for 10 min
     - Warning
   * - UmpireHighMemoryUsage
     - Allocated > 10GB for 5 min
     - Warning
   * - UmpireAllocationStorm
     - Allocation rate > 1000/sec for 2 min
     - Info
   * - UmpireClusterMemoryUsage
     - Total cluster memory > 1TB for 5 min
     - Critical
   * - UmpireNoDeallocationActivity
     - Alloc rate > 0 but dealloc rate == 0 for 15 min
     - Warning
   * - UmpireLargeAllocationSize
     - Average allocation size > 100MB for 5 min
     - Info
   * - UmpireMetricsStale
     - Metrics not updated for 5 min
     - Critical

Troubleshooting
===============

Metrics Not Appearing
---------------------

1. **Check if streaming is enabled**:

   .. code-block:: bash

      echo $UMPIRE_STREAMING_BACKEND  # Should output "prometheus"

2. **Verify Prometheus endpoint is reachable**:

   .. code-block:: bash

      curl -v $UMPIRE_PROMETHEUS_ENDPOINT

3. **Check application logs** for streaming backend errors:

   .. code-block:: bash

      grep "prometheus_backend" application.log

4. **Verify Prometheus is scraping**:

   Open Prometheus web UI (http://localhost:9090) and check targets.

High Memory Overhead
--------------------

If streaming causes high memory overhead:

1. **Reduce buffer size**:

   .. code-block:: bash

      export UMPIRE_RING_BUFFER_SIZE=8192

2. **Increase flush interval** (sends less frequently):

   .. code-block:: bash

      export UMPIRE_PROMETHEUS_FLUSH_INTERVAL_SEC=60

3. **Disable per-allocator tracking** (not yet supported, future enhancement)

Backend Enters Fallback Mode
-----------------------------

If you see ``prometheus_backend: entering FALLBACK mode`` in logs:

1. **Prometheus is unreachable**: Check network connectivity and firewall rules
2. **Prometheus is overloaded**: Increase flush interval or reduce number of ranks
3. **Check Prometheus logs** for rejected requests:

   .. code-block:: bash

      journalctl -u prometheus -f

When backend enters fallback mode, the application continues running normally
but metrics are no longer sent. File-based replay (if enabled) is unaffected.

Events Dropped (Overflow)
--------------------------

If you see ``streaming_event_sink: overflow_count > 0`` warnings:

1. **Increase buffer size**:

   .. code-block:: bash

      export UMPIRE_RING_BUFFER_SIZE=32768

2. **Check if allocation rate is unusually high**:

   Profile your application to identify allocation hotspots.

3. **Verify background thread is not stalled**:

   Check for thread contention or CPU starvation on the export thread.

Performance Impact
==================

The streaming infrastructure is designed for minimal overhead:

- **Per-event overhead**: ~60-90ns (0.3% at 40k events/sec with 4 threads)
- **Memory overhead**: ~8MB (queue + aggregator state)
- **CPU overhead**: <1% (background thread flushes periodically)
- **Network bandwidth**: ~10KB per flush (every 30s)

.. note::
   Overhead is measured on a 2.5 GHz x86_64 system with 4 allocation threads.
   Actual overhead may vary based on hardware and workload characteristics.

To measure overhead in your environment:

1. **Baseline**: Run application with ``UMPIRE_STREAMING_BACKEND`` unset
2. **With streaming**: Run with streaming enabled
3. **Compare wall-clock time** and peak memory usage

If overhead exceeds 5%, consider:

- Increasing flush interval (less frequent sends)
- Reducing buffer size (lower memory footprint)
- Disabling streaming for performance-critical ranks

Best Practices
==============

Production Deployment
---------------------

1. **Start with a pilot**: Enable streaming on 1-2 ranks first to validate setup
2. **Set reasonable thresholds**: Adjust alert thresholds based on baseline profiling
3. **Monitor Prometheus itself**: Ensure Prometheus has enough resources for your scale
4. **Use remote storage**: For large clusters (>1000 ranks), use Thanos or Cortex
5. **Document runbooks**: Link alert annotations to internal runbooks for faster triage

Development and Testing
-----------------------

1. **Use file replay for debugging**: Streaming is for monitoring, not forensics
2. **Test alert rules**: Use Prometheus' ``promtool`` to validate rules before deployment
3. **Verify dashboard panels**: Ensure queries return expected data before scaling up

Capacity Planning
-----------------

For a cluster with **N ranks** and **R allocation rate per rank**:

- **Prometheus write throughput**: ~N * R / 30 metrics/sec (with 30s flush interval)
- **Prometheus storage**: ~N * 100 KB/hour (depends on cardinality)
- **Network bandwidth**: ~N * 10 KB / 30s (depends on number of allocators)

Example: 1000 ranks with 1000 allocs/sec each:

- Prometheus writes: 1000 * 1000 / 30 = ~33k metrics/sec
- Prometheus storage: 1000 * 100 KB/hour = ~100 MB/hour = ~2.4 GB/day
- Network bandwidth: 1000 * 10 KB / 30s = ~333 KB/sec = 2.7 MB/sec

Limitations
===========

- **No per-allocation tracing**: Use file replay for detailed allocation history
- **Aggregate metrics only**: Cannot query individual pointer addresses
- **Prometheus deployment required**: Streaming is disabled if Prometheus unavailable
- **No historical backfill**: Metrics are only available from when streaming started
- **MPI rank labeling**: Requires MPI-aware builds (``UMPIRE_ENABLE_MPI=On``)

See Also
========

- :ref:`logging_and_replay`: File-based replay for detailed forensics
- :ref:`file_output`: Configuring output file locations
- `Prometheus Documentation <https://prometheus.io/docs/>`_
- `Grafana Documentation <https://grafana.com/docs/>`_

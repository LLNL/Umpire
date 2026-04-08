#!/bin/bash
##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

# Script to run introspection level benchmarks and generate comparison report

BENCHMARK_BIN="${1:-./bin/introspection_level_benchmarks}"

if [ ! -f "$BENCHMARK_BIN" ]; then
    echo "Error: Benchmark binary not found at $BENCHMARK_BIN"
    echo "Usage: $0 [path_to_benchmark_binary]"
    echo "Example: $0 ./build/bin/introspection_level_benchmarks"
    exit 1
fi

OUTPUT_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "Introspection Level Benchmark Suite"
echo "============================================"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Running benchmarks for all three levels..."
echo ""

# Run benchmarks for each introspection level
for LEVEL in off basic on; do
    echo "============================================"
    echo "Running: Level = $LEVEL"
    echo "============================================"

    UMPIRE_INTROSPECTION_LEVEL="$LEVEL" "$BENCHMARK_BIN" \
        --benchmark_out="$OUTPUT_DIR/results_${LEVEL}.json" \
        --benchmark_out_format=json \
        --benchmark_counters_tabular=true \
        | tee "$OUTPUT_DIR/results_${LEVEL}.txt"

    echo ""
done

echo ""
echo "============================================"
echo "Results saved to $OUTPUT_DIR/"
echo "  - results_off.txt/json: Off mode results"
echo "  - results_basic.txt/json: Basic mode results"
echo "  - results_on.txt/json: On mode results"
echo "============================================"
echo ""

# Generate comparison summary
echo "Performance Comparison Summary" | tee "$OUTPUT_DIR/comparison.txt"
echo "==============================" | tee -a "$OUTPUT_DIR/comparison.txt"
echo "" | tee -a "$OUTPUT_DIR/comparison.txt"

echo "ALLOCATION PERFORMANCE (lower is better):" | tee -a "$OUTPUT_DIR/comparison.txt"
echo "-----------------------------------------" | tee -a "$OUTPUT_DIR/comparison.txt"
for LEVEL in off basic on; do
    echo "" | tee -a "$OUTPUT_DIR/comparison.txt"
    echo "[$LEVEL]:" | tee -a "$OUTPUT_DIR/comparison.txt"
    grep "^BM_Allocate " "$OUTPUT_DIR/results_${LEVEL}.txt" 2>/dev/null | head -1 | tee -a "$OUTPUT_DIR/comparison.txt"
done
echo "" | tee -a "$OUTPUT_DIR/comparison.txt"

echo "QUERY PERFORMANCE - hasAllocator (lower is better):" | tee -a "$OUTPUT_DIR/comparison.txt"
echo "---------------------------------------------------" | tee -a "$OUTPUT_DIR/comparison.txt"
for LEVEL in basic on; do
    echo "" | tee -a "$OUTPUT_DIR/comparison.txt"
    echo "[$LEVEL]:" | tee -a "$OUTPUT_DIR/comparison.txt"
    grep "^BM_HasAllocator " "$OUTPUT_DIR/results_${LEVEL}.txt" 2>/dev/null | head -1 | tee -a "$OUTPUT_DIR/comparison.txt"
done
echo "(Off mode: N/A - always returns false)" | tee -a "$OUTPUT_DIR/comparison.txt"
echo "" | tee -a "$OUTPUT_DIR/comparison.txt"

echo "QUERY PERFORMANCE - getAllocator (lower is better):" | tee -a "$OUTPUT_DIR/comparison.txt"
echo "----------------------------------------------------" | tee -a "$OUTPUT_DIR/comparison.txt"
for LEVEL in basic on; do
    echo "" | tee -a "$OUTPUT_DIR/comparison.txt"
    echo "[$LEVEL]:" | tee -a "$OUTPUT_DIR/comparison.txt"
    grep "^BM_GetAllocator " "$OUTPUT_DIR/results_${LEVEL}.txt" 2>/dev/null | head -1 | tee -a "$OUTPUT_DIR/comparison.txt"
done
echo "(Off mode: N/A - throws exception)" | tee -a "$OUTPUT_DIR/comparison.txt"
echo "" | tee -a "$OUTPUT_DIR/comparison.txt"

echo "COPY PERFORMANCE (lower is better):" | tee -a "$OUTPUT_DIR/comparison.txt"
echo "-----------------------------------" | tee -a "$OUTPUT_DIR/comparison.txt"
for LEVEL in basic on; do
    echo "" | tee -a "$OUTPUT_DIR/comparison.txt"
    echo "[$LEVEL]:" | tee -a "$OUTPUT_DIR/comparison.txt"
    grep "^BM_Copy " "$OUTPUT_DIR/results_${LEVEL}.txt" 2>/dev/null | head -1 | tee -a "$OUTPUT_DIR/comparison.txt"
done
echo "(Off mode: N/A - not available)" | tee -a "$OUTPUT_DIR/comparison.txt"
echo "" | tee -a "$OUTPUT_DIR/comparison.txt"

echo "MEMORY OVERHEAD SCALING (time for 100K allocations):" | tee -a "$OUTPUT_DIR/comparison.txt"
echo "-----------------------------------------------------" | tee -a "$OUTPUT_DIR/comparison.txt"
for LEVEL in off basic on; do
    echo "" | tee -a "$OUTPUT_DIR/comparison.txt"
    echo "[$LEVEL]:" | tee -a "$OUTPUT_DIR/comparison.txt"
    grep "^BM_MemoryOverhead/100000 " "$OUTPUT_DIR/results_${LEVEL}.txt" 2>/dev/null | tee -a "$OUTPUT_DIR/comparison.txt"
done
echo "" | tee -a "$OUTPUT_DIR/comparison.txt"

echo "" | tee -a "$OUTPUT_DIR/comparison.txt"
echo "KEY FINDINGS:" | tee -a "$OUTPUT_DIR/comparison.txt"
echo "------------" | tee -a "$OUTPUT_DIR/comparison.txt"
echo "- Off/Basic should have similar allocation performance (no tracking overhead)" | tee -a "$OUTPUT_DIR/comparison.txt"
echo "- On mode has tracking overhead (Judy array insertions)" | tee -a "$OUTPUT_DIR/comparison.txt"
echo "- Basic mode queries are SLOWER (runtime API calls) vs On mode (map lookups)" | tee -a "$OUTPUT_DIR/comparison.txt"
echo "- Basic mode has ZERO storage overhead, On mode has ~24-48 bytes per allocation" | tee -a "$OUTPUT_DIR/comparison.txt"
echo "" | tee -a "$OUTPUT_DIR/comparison.txt"

echo ""
echo "See $OUTPUT_DIR/comparison.txt for side-by-side comparison"

#!/bin/bash
##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set -euo pipefail

replay_tests_dir=$1
testprogram=$replay_tests_dir/replay_api_v2_host_audit

cleanup() {
  rm -f umpire.*.stats umpire.*.replay.bin
}

trap cleanup EXIT
cleanup

UMPIRE_REPLAY="On" "$testprogram"

stats_file=$(ls -1t umpire.*.stats 2>/dev/null | head -n 1)
if [ -z "${stats_file}" ]; then
  echo "Failed: replay audit did not produce a stats file"
  exit 1
fi

segment_count() {
  local start=$1
  local end=$2
  local pattern=$3

  awk -v start="$start" -v end="$end" -v pattern="$pattern" '
    index($0, "\"category\":\"metadata\"") &&
    index($0, "\"name\":\"event\"") &&
    index($0, "\"string_args\":{\"name\":\"" start "\"}") { in_segment=1; next }

    index($0, "\"category\":\"metadata\"") &&
    index($0, "\"name\":\"event\"") &&
    index($0, "\"string_args\":{\"name\":\"" end "\"}") { in_segment=0 }

    in_segment && $0 ~ pattern { count++ }

    END { print count + 0 }
  ' "$stats_file"
}

count_named_op() {
  local start=$1
  local end=$2
  local op_name=$3

  segment_count "$start" "$end" "\"category\":\"operation\".*\"name\":\"${op_name}\""
}

count_operations() {
  local start=$1
  local end=$2

  segment_count "$start" "$end" "\"category\":\"operation\""
}

assert_eq() {
  local expected=$1
  local actual=$2
  local message=$3

  if [ "$expected" -ne "$actual" ]; then
    echo "Replay audit failed: ${message} (expected ${expected}, got ${actual})"
    exit 1
  fi
}

if ! grep -q '"name":"make_allocator".*"allocator_name":"API_V2_REPLAY_MOVE_DEST"' "$stats_file"; then
  echo "Replay audit failed: missing make_allocator event for API_V2_REPLAY_MOVE_DEST"
  exit 1
fi

assert_eq 1 "$(count_named_op v1_control_begin v1_control_end allocate)" "v1 control should emit one allocate event"
assert_eq 1 "$(count_named_op v1_control_begin v1_control_end deallocate)" "v1 control should emit one deallocate event"
assert_eq 0 "$(count_named_op v1_control_begin v1_control_end copy)" "v1 control should not emit a copy event"
assert_eq 0 "$(count_named_op v1_control_begin v1_control_end reallocate)" "v1 control should not emit a reallocate event"
assert_eq 0 "$(count_named_op v1_control_begin v1_control_end move)" "v1 control should not emit a move event"

assert_eq 1 "$(count_named_op v2_direct_begin v2_direct_end allocate)" "direct v2 host allocation lifecycle should emit one allocate event"
assert_eq 1 "$(count_named_op v2_direct_begin v2_direct_end deallocate)" "direct v2 host allocation lifecycle should emit one deallocate event"
assert_eq 2 "$(count_operations v2_direct_begin v2_direct_end)" "direct v2 host allocation lifecycle should only emit allocate and deallocate"

assert_eq 1 "$(count_named_op v2_rm_deallocate_begin v2_rm_deallocate_end allocate)" "rm.deallocate on a v2-backed host allocation should preserve the source allocate event"
assert_eq 1 "$(count_named_op v2_rm_deallocate_begin v2_rm_deallocate_end deallocate)" "rm.deallocate on a v2-backed host allocation should preserve the source deallocate event"
assert_eq 2 "$(count_operations v2_rm_deallocate_begin v2_rm_deallocate_end)" "rm.deallocate on a v2-backed host allocation should only emit allocate and deallocate"

assert_eq 2 "$(count_named_op v2_rm_copy_begin v2_rm_copy_end allocate)" "copy audit should only log source and destination allocation lifecycles"
assert_eq 2 "$(count_named_op v2_rm_copy_begin v2_rm_copy_end deallocate)" "copy audit should only log source and destination deallocation lifecycles"
assert_eq 0 "$(count_named_op v2_rm_copy_begin v2_rm_copy_end copy)" "copy audit should not emit a copy event for a v2-backed source allocation"
assert_eq 4 "$(count_operations v2_rm_copy_begin v2_rm_copy_end)" "copy audit should only emit allocate and deallocate lifecycle events"

assert_eq 2 "$(count_named_op v2_rm_reallocate_begin v2_rm_reallocate_end allocate)" "reallocate audit should log old and new allocation lifecycles"
assert_eq 2 "$(count_named_op v2_rm_reallocate_begin v2_rm_reallocate_end deallocate)" "reallocate audit should log old and new deallocation lifecycles"
assert_eq 0 "$(count_named_op v2_rm_reallocate_begin v2_rm_reallocate_end reallocate)" "reallocate audit should not emit a reallocate event for a v2-backed host allocation"
assert_eq 4 "$(count_operations v2_rm_reallocate_begin v2_rm_reallocate_end)" "reallocate audit should only emit allocate and deallocate lifecycle events"

assert_eq 2 "$(count_named_op v2_rm_move_begin v2_rm_move_end allocate)" "move audit should log source and destination allocation lifecycles"
assert_eq 2 "$(count_named_op v2_rm_move_begin v2_rm_move_end deallocate)" "move audit should log source and destination deallocation lifecycles"
assert_eq 0 "$(count_named_op v2_rm_move_begin v2_rm_move_end move)" "move audit should not emit a move event for a v2-backed source allocation"
assert_eq 4 "$(count_operations v2_rm_move_begin v2_rm_move_end)" "move audit should only emit allocate and deallocate lifecycle events"

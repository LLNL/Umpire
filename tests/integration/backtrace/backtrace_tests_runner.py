##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

import sys
import subprocess
import os

errors = 0

def compare(good_output, test_output):
    global errors

    index = 1
    for good_line in good_output[1:]:
        good_words = good_line.split()
        test_words = test_output[index].split()
        index = index + 1
        if len(good_words) >= 2:
            if good_words[2].split('(')[0] != test_words[2].split('(')[0]:
                print 'MISCOMPARE LINE: {} "{}" != "{}"'.format(index+1, good_words[2].split('(')[0], test_words[2].split('(')[0])
                errors = -1

def run_test():
    test_output=subprocess.check_output("./backtrace_tests; exit 0",
            stderr=subprocess.STDOUT,
            shell=True).splitlines()

    good_output_file = sys.argv[1]
    with open(good_output_file, 'r') as good_file:
        good_output = good_file.read().splitlines()

    compare(good_output, test_output)

if __name__ == '__main__':
    run_test()
    sys.exit(errors)

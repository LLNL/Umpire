##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

formatters = {
    'RED': '\033[91m',
    'BLUE': '\033[94m',
    'END': '\033[0m',
}

errors = 0

def check_output(file_object, expected):
    global errors

    print("{BLUE}[RUN     ]{END} Checking for \"{expected}\"".format(expected=expected, **formatters))

    found = False

    for line in file_object:
        if expected in line.rstrip():
            found = True

    if not found:
        print("{RED}[   ERROR]{END} Got {contents}".format(contents=file_object.read(), **formatters))
        errors = errors + 1
    else:
        print("{BLUE}[      OK]{END} Found \"{expected}\"".format(expected=expected, **formatters))


def run_sanitizer_test(strategy, kind):
    import subprocess
    import os

    cmd_args = ['./sanitizer_tests']
    cmd_args.append(strategy)
    cmd_args.append(kind)

    test_program = subprocess.Popen(cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False)
    pid = test_program.pid
    test_program.wait()

    output = test_program.stdout
    error = test_program.stderr

    expected_string = '{} of size 8'.format(kind.upper())
    check_output(error, expected_string)


if __name__ == '__main__':
    import sys

    print("{BLUE}[--------]{END}".format(**formatters))
    run_sanitizer_test('DynamicPoolList', 'read')
    run_sanitizer_test('DynamicPoolList', 'write')
    run_sanitizer_test('QuickPool', 'read')
    run_sanitizer_test('QuickPool', 'write')
    print("{BLUE}[--------]{END}".format(**formatters))
    sys.exit(errors)

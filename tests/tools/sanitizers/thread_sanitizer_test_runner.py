##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
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
    contents = file_object.read()

    if expected in contents:
        found = True

    if not found:
        print("{RED}[   ERROR]{END} Got {contents}".format(contents=contents.decode(errors='replace'),
                                                           **formatters))
        errors = errors + 1
    else:
        print("{BLUE}[      OK]{END} Found \"{expected}\"".format(expected=expected, **formatters))


def run_thread_sanitizer_test(mode):
    import subprocess

    cmd_args = ['./thread_sanitizer_tests', mode]

    test_program = subprocess.Popen(cmd_args,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    shell=False)
    test_program.wait()

    expected_string = b'ThreadSanitizer: data race'
    check_output(test_program.stderr, expected_string)


if __name__ == '__main__':
    import sys

    print("{BLUE}[--------]{END}".format(**formatters))
    run_thread_sanitizer_test('monotonic_buffer')
    print("{BLUE}[--------]{END}".format(**formatters))
    sys.exit(errors)

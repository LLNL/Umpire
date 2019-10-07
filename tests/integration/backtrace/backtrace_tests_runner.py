##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
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

def check_output(name, file_object, expected):
    global errors

    print("{BLUE}[RUN     ]{END} Checking for \"{expected}\" in {name}".format(name=name, expected=expected, **formatters))

    found = False

    for line in file_object:
        if expected in line:
            found = True

    if not found:
        print("{RED}[   ERROR]{END} Got {contents}".format(contents=file_object.read(), expected=expected, **formatters))
        errors = errors + 1
    else:
        print("{BLUE}[      OK]{END} Found \"{expected}\" in {name}".format(name=name, expected=expected, **formatters))


def check_file_exists(filename):
    import os

    global errors

    print("{BLUE}[RUN     ]{END} Checking {myfile} exists".format(myfile=filename, **formatters))
    if not os.path.isfile(filename):
        print("{RED}[   ERROR]{END} {myfile} not found".format(myfile=filename, **formatters))
        errors += errors + 1
    else:
        print("{BLUE}[      OK]{END} {myfile} exists".format(myfile=filename, **formatters))


def run_test():
    import subprocess
    import os

    output=subprocess.check_output("./backtrace_tests; exit 0",
            stderr=subprocess.STDOUT,
            shell=True)

    print output

if __name__ == '__main__':
    import sys

    print("{BLUE}[--------]{END}".format(**formatters))
    run_test()
    print("{BLUE}[--------]{END}".format(**formatters))
    sys.exit(errors)

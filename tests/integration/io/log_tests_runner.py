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

    test_env = {"UMPIRE_LOG_LEVEL" : "Debug"}
    cmd_args = ['./log_tests']
    test_program = subprocess.Popen(cmd_args,
            env=dict(os.environ, **test_env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False)
    pid = test_program.pid
    test_program.wait()

    file_uid = 0;

    output = test_program.stdout

    check_output('stdout', output, "initialize Umpire")

    output_filename = 'umpire.{pid}.{uid}.log'.format(uid=file_uid, pid=pid)

    check_file_exists(output_filename)
    with open(output_filename) as output_file:
        check_output(output_filename, output_file, 'initialize Umpire')

if __name__ == '__main__':
    import sys

    print("{BLUE}[--------]{END}".format(**formatters))
    run_test()
    print("{BLUE}[--------]{END}".format(**formatters))
    sys.exit(errors)

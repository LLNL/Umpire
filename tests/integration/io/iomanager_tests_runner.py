##############################################################################
# Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
#
# Created by David Beckingsale, david@llnl.gov
# LLNL-CODE-747640
#
# All rights reserved.
#
# This file is part of Umpire.
#
# For details, see https://github.com/LLNL/Umpire
# Please also see the LICENSE file for MIT license.
##############################################################################

formatters = {
    'RED': '\033[91m',
    'GREEN': '\033[92m',
    'END': '\033[0m',
}


errors = 0


def check_output(name, file_object, expected):

    print("{GREEN}[RUN     ]{END} Checking for \"{expected}\" in {name}".format(name=name, expected=expected, **formatters))

    contents = file_object.readline().rstrip()
    if (contents != expected):
        print("{RED}[   ERROR]){END} Got {contents}".format(contents=contents, expected=expected, **formatters))
        errors += 1
    else:
        print("{GREEN}[      OK]{END} Found \"{expected}\" in {name}".format(name=name, expected=expected, **formatters))


def check_file_exists(filename):
    import os

    print("{GREEN}[RUN     ]{END} Checking {myfile} exists".format(myfile=filename, **formatters))
    if (not os.path.isfile(filename)):
        print("{RED}[   ERROR]){END} {myfile} not found".format(myfile=filename, **formatters))
        errors += 1
    else:
        print("{GREEN}[      OK]{END} {myfile} exists".format(myfile=filename, **formatters))


def run_io_test(test_env, file_uid):
    import subprocess
    import os

    test_program = subprocess.Popen('./iomanager_tests', 
            env=dict(os.environ, **test_env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

    test_program.wait()

    output = test_program.stdout
    error = test_program.stderr

    check_output('stdout', output, 'testing log stream')
    check_output('stderr', error, 'testing error stream')

    output_filename = 'umpire_io_tests.0.{uid}.log'.format(uid=file_uid)
    replay_filename = 'umpire_io_tests.0.{uid}.replay'.format(uid=file_uid)
    if 'UMPIRE_OUTPUT_DIR' in test_env.keys():
        output_filename = '{dir}/umpire_io_tests.0.{uid}.log'.format(dir=test_env['UMPIRE_OUTPUT_DIR'], uid=file_uid)
        replay_filename = '{dir}/umpire_io_tests.0.{uid}.replay'.format(dir=test_env['UMPIRE_OUTPUT_DIR'], uid=file_uid)

    check_file_exists(output_filename)
    check_file_exists(replay_filename)

    with open(output_filename) as output_file:
        check_output(output_filename, output_file, 'testing log stream')

    with open(replay_filename) as replay_file:
        check_output(replay_filename, replay_file, 'testing replay stream')



if __name__ == '__main__':
    import sys
    print("{GREEN}[--------]{END}".format(**formatters))
    run_io_test({'UMPIRE_OUTPUT_BASENAME' : 'umpire_io_tests'}, 0)
    run_io_test({'UMPIRE_OUTPUT_DIR': './io_test_dir', 'UMPIRE_OUTPUT_BASENAME' : 'umpire_io_tests'}, 0)
    run_io_test({'UMPIRE_OUTPUT_DIR': './io_test_dir', 'UMPIRE_OUTPUT_BASENAME' : 'umpire_io_tests'}, 1)
    print("{GREEN}[--------]{END}".format(**formatters))
    sys.exit(errors)

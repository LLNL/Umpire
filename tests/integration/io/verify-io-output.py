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

def check_output(file_object, expected):
    import sys

    contents = file_object.readline().rstrip()
    if (contents != expected):
        print("ERROR: output incorrect. Was %s, expected \"%s\"" % (contents, expected))
        sys.exit(1)


def run_io_test():
    import subprocess
    import os
    import sys

    print(os.getcwd())

    test_env = {'UMPIRE_OUTPUT_BASENAME' : 'umpire_io_tests'}

    test_program = subprocess.Popen('./iomanager_tests', 
            env=dict(os.environ, **test_env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

    test_program.wait()

    output = test_program.stdout
    error = test_program.stderr

    check_output(output, 'testing log stream')
    check_output(error, 'testing error stream')

    output_filename = 'umpire_io_tests.0.0.log'
    replay_filename = 'umpire_io_tests.0.0.replay'

    if (not os.path.isfile(output_filename)):
        print("ERROR: %s doesn't exist" % (output_filename))
        sys.exit(1)

    if (not os.path.isfile(replay_filename)):
        print("ERROR: %s doesn't exist" % (replay_filename))
        sys.exit(1)

    with open(output_filename) as output_file:
        check_output(output_file, 'testing log stream')

    with open(replay_filename) as replay_file:
        check_output(replay_file, 'testing replay stream')

if __name__ == '__main__':
    run_io_test()

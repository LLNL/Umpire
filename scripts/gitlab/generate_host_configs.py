#!/usr/bin/env python3

import json
import os
import subprocess
from string import digits
import sys
import time

from optparse import OptionParser

def cmd_exe(cmd,ret_output=False,echo=False):
    """
    Helper for shell commands execution.
    """
    if echo:
        print("[cmd_exe: {}]".format(cmd))
    if ret_output:
        p = subprocess.Popen(cmd,
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        out = p.communicate()[0]
        out = out.decode('utf8')
        return p.returncode,out
    else:
        return subprocess.call(cmd,shell=True)


def get_machine_name():
    """
    Use hostname and strip digit to get machine name
    """
    cmd = "hostname"
    res, out = cmd_exe(cmd, ret_output=True)

    if res != 0:
        print("[ERROR]: '{}' call failed".format(cmd))
        sys.exit(-1)

    hostname=out.strip().strip(digits)

    if not hostname:
        print("[ERROR]: hostname is empty")
        sys.exit(-1)

    return hostname


def get_system_type():
    """
    Use environment variable to get system type
    """
    sys_type = os.environ.get('SYS_TYPE')

    if not sys_type:
        print("[ERROR]: sys_type is empty")
        sys.exit(-1)

    return sys_type


def parse_args():
    """
    Parses args from command line
    """
    parser = OptionParser()
    parser.add_option("--spec-filter",
                      dest="spec-filter",
                      default=None,
                      help="Partial spec to match")

    parser.add_option("--forced-spec",
                      dest="forced-spec",
                      default=False,
                      help="Force the use of this spec without checking spec list")

    parser.add_option("--shm",
                      dest="use_shm",
                      default=False,
                      help="Use Spack in shared memory")

    opts, extras = parser.parse_args()
    opts = vars(opts)

    return opts, extras


# Main
def main():
    """
    Generate host-config files for all specs matching sys_type, machine and 
    partial_spec. Specs for each (sys_type,machine) are provided in a json 
    formated file.
    """

    opts, extra_opts = parse_args()

    partial_spec = opts["spec-filter"]
    exact_spec = opts["forced-spec"]
    machine = get_machine_name()
    sys_type = get_system_type()

    uberenv_cmd_opts = ""

    if opts["use_shm"]:
        prefix="/dev/shm/uberenv_libs-{0}-{1}-{2}".format(sys_type,machine,time.time())
        uberenv_cmd_opts = ' '.join([uberenv_cmd_opts,"--prefix={0}".format(prefix)])
        print("[INFO]: Uberenv will use prefix {0}".format(prefix))

    with open('scripts/gitlab/list_of_specs.json') as f:
        specs_data = json.load(f)

        if exact_spec:
            print("[INFO]: generate host-config for spec {0}".format(exact_spec))
            cmd = "python scripts/uberenv/uberenv.py --spec={0} {1}".format(exact_spec,uberenv_cmd_opts)
            cmd_exe(cmd, echo=True)
        else:
            # Exclude spec not matching partial_spec if defined
            for spec in specs_data[sys_type][machine]:
                if partial_spec and not partial_spec in spec:
                    print("[INFO]: spec {0} ignored".format(spec))
                else:
                    print("[INFO]: generate host-config for spec {0}".format(spec))
                    cmd = "python scripts/uberenv/uberenv.py --spec={0} {1}".format(spec,uberenv_cmd_opts)
                    cmd_exe(cmd, echo=True)

if __name__ == "__main__":
    sys.exit(main())

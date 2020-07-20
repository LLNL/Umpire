#!/usr/bin/env python3

import json
import os
import subprocess
from string import digits
import sys

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
    parser.add_option("--spec",
                      dest="spec",
                      default=None,
                      help="Partial spec to match")

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
    
    partial_spec = opts["spec"]
    machine = get_machine_name()
    sys_type = get_system_type()
    
    with open('scripts/gitlab/list_of_specs.json') as f:
        specs_data = json.load(f)
    
        # Exclude spec not matching partial_spec if defined
        for spec in specs_data[sys_type][machine]:
            if partial_spec and not partial_spec in spec:
                print("[INFO]: spec {0} ignored".format(spec))
            else:
                print("[INFO]: generate host-config for spec {0}".format(spec))
                cmd = "python scripts/uberenv/uberenv.py --spec={0}".format(spec)
                cmd_exe(cmd, echo=True)

if __name__ == "__main__":
    sys.exit(main())

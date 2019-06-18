.. _file_output:

========
File I/O
========

Umpire provides support for writing files containing log and replay data,
rather than directing this output to stdout. When logging or replay are
enabled, the following environment variables can be used to determine where the
output is written:


      ===========================  ======== ===============================================================================
      Variable                     Default  Description
      ===========================  ======== ===============================================================================
      ``UMPIRE_OUTPUT_DIR``        .        Directory to write log and replay files
      ``UMPIRE_OUTPUT_BASENAME``   umpire   Basename of logging and relpay files


The values of these variables are used to construct unique filenames for
output. The extension ``.log`` is used for logging output, and ``.replay`` for
replay output. The filenames additionally contain two integers, one
corresponding to the rank of the process, and one that is used to make multiple
files with the same basename and rank unique. This ensures that multiple runs
with the same IO configuration do not overwrite files.

The format of the filenames is:

.. code-block:: bash

    <UMPIRE_OUTPUT_BASENAME>.<RANK>.<UID>.<log|replay>

If Umpire is compiled without MPI support, then rank will always be 0.

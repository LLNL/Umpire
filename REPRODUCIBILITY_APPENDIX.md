# Reproducibility Information

The actual data and replay file that we used in our experiments is sensitive information from an LLNL internal multi-physics application which can not be shared publicly. However, these experiments can be replicated with non-sensitive data gathered from running the Replay tool with any application which uses Umpire for memory management. In order to reproduce our results with your own application, you can follow the below sequence of steps.

- Clone the Umpire repository on Github.

   $ git clone https://github.com/LLNL/Umpire.git

- Switch to the correct branch.

   $ git checkout poster_experimental_2022

- Pull in Umpire submodules and build with the Replay tool enabled.

   $ git submodule init && git submodule update
   $ mkdir build
   $ cd build
   $ cmake -DENABLE_TOOLS=On ../
   $ make -j

  Note: You can also change the default compiler used to build Umpire by changing the -DCMAKE_CXX_COMPILER=/path/to/compiler/ and -DCMAKE_C_COMPILER=/path/to/compiler/ cmake commands. If your program requires device memory, be use to set -DENABLE_CUDA=On.

- Edit the pool source code with the heuristic option you'd like.

   $ vi ../src/umpire/strategy/Quickpool.cpp

  On lines 16-26, you will see commented out #ifdef include options to turn on or off the USE_HIGHWATERMARK and COALESCE_BEFORE_GROW options. Uncomment the option you'd like to turn on.

  Although we edited the QuickPool source code for these specific experiments, similar edits could also be made for the other pools that Umpire provides. Rebuild the code with make.

- Run the replay tool with some code that uses \verb|QuickPool|.

   $ UMPIRE_REPLAY="On" ./my-umpire-program

  This will create a \verb|.replay| file which you can "replay" with the Replay tool later.
  Note: More information about running with Umpire's Replay tool can be found on Umpire's Replay documentation\footnote{https://umpire.readthedocs.io/en/develop/sphinx/features/logging\_and\_replay.html}.

- Rerun your new replay file with the Umpire Replay tool.
  The below example will run a replay file with QuickPool, using a coalescing heuristic of Blocks-Releasable, and heuristic parameter of 2.
  It will then display stats about the replay run (-s), show timing information (-t), and dump out a .ult file (-d).

   $ ./bin/replay -i my-umpire-program.replay --use-pool Quick --use-heuristic Block --heuristic-parm 2 --recompile -s -t -d

  To see helpful information about the replay tool, you can do the following:

   $ ./bin/replay --help

  This will print out a verbose list of parameters you can run with the Replay tool.

- Analyze the results. If you have pydv, you can open the resulting .ult file and plot the results.

   $ /path/to/pydv/pdv my-umpire-program.ult

  The [pydv documentation](https://pydv.readthedocs.io/en/latest/index.html) describes the commands you can use to query the information gathered about the pools used in the program.

If there are questions, comments, and/or concerns, please reach out to the Umpire team at umpire-dev@llnl.gov.

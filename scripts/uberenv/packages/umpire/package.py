# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


from spack import *

import socket
import os

from os import environ as env
from os.path import join as pjoin

import re

def cmake_cache_entry(name, value, comment=""):
    """Generate a string for a cmake cache variable"""

    return 'set(%s "%s" CACHE PATH "%s")\n\n' % (name,value,comment)


def cmake_cache_string(name, string, comment=""):
    """Generate a string for a cmake cache variable"""

    return 'set(%s "%s" CACHE STRING "%s")\n\n' % (name,string,comment)


def cmake_cache_option(name, boolean_value, comment=""):
    """Generate a string for a cmake configuration option"""

    value = "ON" if boolean_value else "OFF"
    return 'set(%s %s CACHE BOOL "%s")\n\n' % (name,value,comment)


def get_spec_path(spec, package_name, path_replacements = {}, use_bin = False) :
    """Extracts the prefix path for the given spack package
       path_replacements is a dictionary with string replacements for the path.
    """

    if not use_bin:
        path = spec[package_name].prefix
    else:
        path = spec[package_name].prefix.bin

    path = os.path.realpath(path)

    for key in path_replacements:
        path = path.replace(key, path_replacements[key])

    return path

class Umpire(CMakePackage, CudaPackage):
    """An application-focused API for memory management on NUMA & GPU
    architectures"""

    homepage = 'https://github.com/LLNL/Umpire'
    git      = 'https://github.com/LLNL/Umpire.git'

    version('develop', branch='develop', submodules='True')
    version('master', branch='main', submodules='True')
    version('3.0.0', tag='v3.0.0', submodules='True')
    version('2.1.0', tag='v2.1.0', submodules='True')
    version('2.0.0', tag='v2.0.0', submodules='True')
    version('1.1.0', tag='v1.1.0', submodules='True')
    version('1.0.1', tag='v1.0.1', submodules='True')
    version('1.0.0', tag='v1.0.0', submodules='True')
    version('0.3.5', tag='v0.3.5', submodules='True')
    version('0.3.4', tag='v0.3.4', submodules='True')
    version('0.3.3', tag='v0.3.3', submodules='True')
    version('0.3.2', tag='v0.3.2', submodules='True')
    version('0.3.1', tag='v0.3.1', submodules='True')
    version('0.3.0', tag='v0.3.0', submodules='True')
    version('0.2.4', tag='v0.2.4', submodules='True')
    version('0.2.3', tag='v0.2.3', submodules='True')
    version('0.2.2', tag='v0.2.2', submodules='True')
    version('0.2.1', tag='v0.2.1', submodules='True')
    version('0.2.0', tag='v0.2.0', submodules='True')
    version('0.1.4', tag='v0.1.4', submodules='True')
    version('0.1.3', tag='v0.1.3', submodules='True')

    patch('camp_target_umpire_3.0.0.patch', when='@3.0.0')

    variant('fortran', default=False, description='Build C/Fortran API')
    variant('c', default=True, description='Build C API')
    variant('mpi', default=False, description='Enable MPI support')
    variant('numa', default=False, description='Enable NUMA support')
    variant('shared', default=False, description='Enable Shared libs')
    variant('openmp', default=False, description='Build with OpenMP support')
    variant('openmp_target', default=False, description='Build with OpenMP 4.5 support')
    variant('deviceconst', default=False,
            description='Enables support for constant device memory')
    variant('tests', default='basic', values=('none', 'basic', 'benchmarks'),
            multi=False, description='Tests to run')

    variant('libcpp', default=False, description='Uses libc++ instead of libstdc++')
    variant('hip', default=False, description='Build with HIP support')
    variant('tools', default=True, description='Enable tools')
    variant('werror', default=True, description='Enable warnings as errors')
    variant('sanitizer_tests', default=False, description='Enable address sanitizer tests')

    depends_on('cmake@3.8:', type='build')
    depends_on('cmake@3.9:', when='+cuda', type='build')
    depends_on('mpi', when='+mpi')
    depends_on('hip', when='+hip')

    conflicts('+numa', when='@:0.3.2')
    conflicts('~c', when='+fortran', msg='Fortran API requires C API')
    conflicts('~openmp', when='+openmp_target', msg='OpenMP target requires OpenMP')
    conflicts('+cuda', when='+hip')
    conflicts('+openmp', when='+hip')
    conflicts('+openmp_target', when='+hip')
    conflicts('+deviceconst', when='~hip~cuda')

    phases = ['hostconfig', 'cmake', 'build', 'install']

    def _get_sys_type(self, spec):
        sys_type = str(spec.architecture)
        # if on llnl systems, we can use the SYS_TYPE
        if "SYS_TYPE" in env:
            sys_type = env["SYS_TYPE"]
        return sys_type

    def _get_host_config_path(self, spec):
        var=''
        if '+cuda' in spec:
            var= '-'.join([var,'cuda'])
        if '+libcpp' in spec:
            var='-'.join([var,'libcpp'])

        host_config_path = "hc-%s-%s-%s%s-%s.cmake" % (socket.gethostname().rstrip('1234567890'),
                                               self._get_sys_type(spec),
                                               spec.compiler,
                                               var,
                                               spec.dag_hash())
        dest_dir = self.stage.source_path
        host_config_path = os.path.abspath(pjoin(dest_dir, host_config_path))
        return host_config_path

    def hostconfig(self, spec, prefix, py_site_pkgs_dir=None):
        """
        This method creates a 'host-config' file that specifies
        all of the options used to configure and build Umpire.

        For more details about 'host-config' files see:
            http://software.llnl.gov/conduit/building.html

        Note:
          The `py_site_pkgs_dir` arg exists to allow a package that
          subclasses this package provide a specific site packages
          dir when calling this function. `py_site_pkgs_dir` should
          be an absolute path or `None`.

          This is necessary because the spack `site_packages_dir`
          var will not exist in the base class. For more details
          on this issue see: https://github.com/spack/spack/issues/6261
        """

        #######################
        # Compiler Info
        #######################
        c_compiler = env["SPACK_CC"]
        cpp_compiler = env["SPACK_CXX"]

        # Even though we don't have fortran code in our project we sometimes
        # use the Fortran compiler to determine which libstdc++ to use
        f_compiler = ""
        if "SPACK_FC" in env.keys():
            # even if this is set, it may not exist
            # do one more sanity check
            if os.path.isfile(env["SPACK_FC"]):
                f_compiler = env["SPACK_FC"]


        #######################################################################
        # By directly fetching the names of the actual compilers we appear
        # to doing something evil here, but this is necessary to create a
        # 'host config' file that works outside of the spack install env.
        #######################################################################

        sys_type = self._get_sys_type(spec)

        ##############################################
        # Find and record what CMake is used
        ##############################################

        cmake_exe = spec['cmake'].command.path
        cmake_exe = os.path.realpath(cmake_exe)

        host_config_path = self._get_host_config_path(spec)
        cfg = open(host_config_path, "w")
        cfg.write("###################\n".format("#" * 60))
        cfg.write("# Generated host-config - Edit at own risk!\n")
        cfg.write("###################\n".format("#" * 60))
        cfg.write("# Copyright (c) 2020, Lawrence Livermore National Security, LLC and\n")
        cfg.write("# other Umpire Project Developers. See the top-level LICENSE file for\n")
        cfg.write("# details.\n")
        cfg.write("#\n")
        cfg.write("# SPDX-License-Identifier: (BSD-3-Clause) \n")
        cfg.write("###################\n\n".format("#" * 60))

        cfg.write("#------------------\n".format("-" * 60))
        cfg.write("# SYS_TYPE: {0}\n".format(sys_type))
        cfg.write("# Compiler Spec: {0}\n".format(spec.compiler))
        cfg.write("# CMake executable path: %s\n" % cmake_exe)
        cfg.write("#------------------\n\n".format("-" * 60))

        cfg.write(cmake_cache_string("CMAKE_BUILD_TYPE", spec.variants['build_type'].value))

        #######################
        # Compiler Settings
        #######################

        cfg.write("#------------------\n".format("-" * 60))
        cfg.write("# Compilers\n")
        cfg.write("#------------------\n\n".format("-" * 60))
        cfg.write(cmake_cache_entry("CMAKE_C_COMPILER", c_compiler))
        cfg.write(cmake_cache_entry("CMAKE_CXX_COMPILER", cpp_compiler))
        if '+fortran' in spec:
          cfg.write(cmake_cache_entry("CMAKE_Fortran_COMPILER", f_compiler))

        # use global spack compiler flags
        cflags = ' '.join(spec.compiler_flags['cflags'])
        if "+libcpp" in spec:
            cflags += ' '.join([cflags,"-DGTEST_HAS_CXXABI_H_=0"])
        if cflags:
            cfg.write(cmake_cache_entry("CMAKE_C_FLAGS", cflags))

        cxxflags = ' '.join(spec.compiler_flags['cxxflags'])
        if "+libcpp" in spec:
            cxxflags += ' '.join([cxxflags,"-stdlib=libc++ -DGTEST_HAS_CXXABI_H_=0"])
        if cxxflags:
            cfg.write(cmake_cache_entry("CMAKE_CXX_FLAGS", cxxflags))

        fflags = ' '.join(spec.compiler_flags['fflags'])
        cfg.write(cmake_cache_entry("CMAKE_Fortran_FLAGS", fflags))

        fortran_compilers = ["gfortran", "xlf"]
        if any(compiler in f_compiler for compiler in fortran_compilers) and ("clang" in cpp_compiler):
            cfg.write(cmake_cache_entry("BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE",
            "/usr/tce/packages/gcc/gcc-4.9.3/lib64;/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3;/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64;/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/x86_64-unknown-linux-gnu/4.9.3"))

            libdir = pjoin(os.path.dirname(
                           os.path.dirname(f_compiler)), "lib")
            flags = ""
            for _libpath in [libdir, libdir + "64"]:
                if os.path.exists(_libpath):
                    flags += " -Wl,-rpath,{0}".format(_libpath)
            description = ("Adds a missing libstdc++ rpath")
            if flags:
                cfg.write(cmake_cache_entry("BLT_EXE_LINKER_FLAGS", flags,
                                            description))


        gcc_toolchain_regex = re.compile(".*gcc-toolchain.*")
        gcc_name_regex = re.compile(".*gcc-name.*")

        using_toolchain = list(filter(gcc_toolchain_regex.match, spec.compiler_flags['cxxflags']))
        using_gcc_name = list(filter(gcc_name_regex.match, spec.compiler_flags['cxxflags']))
        compilers_using_toolchain = ["pgi", "xl", "icpc"]
        if any(compiler in cpp_compiler for compiler in compilers_using_toolchain):
            if using_toolchain or using_gcc_name:
                cfg.write(cmake_cache_entry("BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE",
                "/usr/tce/packages/gcc/gcc-4.9.3/lib64;/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3;/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64;/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/x86_64-unknown-linux-gnu/4.9.3"))

        if "toss_3_x86_64_ib" in sys_type:
            release_flags = "-O3"
            reldebinf_flags = "-O3 -g"
            debug_flags = "-O0 -g"

            if "intel" in str(spec.compiler):
                release_flags = ' '.join([release_flags,'-finline-functions -axCORE-AVX2 -diag-disable cpu-dispatch'])
                reldebinf_flags = ' '.join([reldebinf_flags,'-finline-functions -axCORE-AVX2 -diag-disable cpu-dispatch'])

            cfg.write(cmake_cache_entry("CMAKE_CXX_FLAGS_RELEASE", release_flags))
            cfg.write(cmake_cache_entry("CMAKE_CXX_FLAGS_RELWITHDEBINFO", reldebinf_flags))
            cfg.write(cmake_cache_entry("CMAKE_CXX_FLAGS_DEBUG", debug_flags))

        if "+cuda" in spec:
            cfg.write("#------------------{0}\n".format("-" * 60))
            cfg.write("# Cuda\n")
            cfg.write("#------------------{0}\n\n".format("-" * 60))

            cfg.write(cmake_cache_option("ENABLE_CUDA", True))

            cudatoolkitdir = spec['cuda'].prefix
            cfg.write(cmake_cache_entry("CUDA_TOOLKIT_ROOT_DIR",
                                        cudatoolkitdir))
            cudacompiler = "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc"
            cfg.write(cmake_cache_entry("CMAKE_CUDA_COMPILER",
                                        cudacompiler))

            cuda_flags = []

            if not spec.satisfies('cuda_arch=none'):
                cuda_arch = spec.variants['cuda_arch'].value
                cuda_flags.append('-arch sm_{0}'.format(cuda_arch[0]))

            if '+deviceconst' in spec:
                cfg.write(cmake_cache_option("ENABLE_DEVICE_CONST", True))

            if using_toolchain:
                cuda_flags.append("-Xcompiler {}".format(using_toolchain[0]))

            cfg.write(cmake_cache_string("CMAKE_CUDA_FLAGS",  ' '.join(cuda_flags)))

        else:
            cfg.write(cmake_cache_option("ENABLE_CUDA", False))

        if "+hip" in spec:
            cfg.write("#------------------{0}\n".format("-" * 60))
            cfg.write("# HIP\n")
            cfg.write("#------------------{0}\n\n".format("-" * 60))

            cfg.write(cmake_cache_option("ENABLE_HIP", True))

#            -DHIP_ROOT_DIR=/opt/rocm-3.6.0/hip -DHIP_CLANG_PATH=/opt/rocm-3.6.0/llvm/bin

            hip_root = spec['hip'].prefix
            rocm_root = hip_root + "/.."
            cfg.write(cmake_cache_entry("HIP_ROOT_DIR",
                                        hip_root))
            cfg.write(cmake_cache_entry("HIP_CLANG_PATH",
                                        rocm_root + '/llvm/bin'))
            cfg.write(cmake_cache_entry("HIP_HIPCC_FLAGS",
                                        '--amdgpu-target=gfx906'))
            cfg.write(cmake_cache_entry("HIP_RUNTIME_INCLUDE_DIRS",
                                        "{0}/include;{0}/../hsa/include".format(hip_root)))
            if '%gcc' in spec:
                gcc_bin = os.path.dirname(self.compiler.cxx)
                gcc_prefix = join_path(gcc_bin, '..')
                cfg.write(cmake_cache_entry("HIP_CLANG_FLAGS",
                "--gcc-toolchain={0}".format(gcc_prefix))) 
                cfg.write(cmake_cache_entry("CMAKE_EXE_LINKER_FLAGS",
                "-Wl,-rpath {}/lib64".format(gcc_prefix)))

            if '+deviceconst' in spec:
                cfg.write(cmake_cache_option("ENABLE_DEVICE_CONST", True))

        else:
            cfg.write(cmake_cache_option("ENABLE_HIP", False))

        cfg.write(cmake_cache_option("ENABLE_C", '+c' in spec))
        cfg.write(cmake_cache_option("ENABLE_FORTRAN", '+fortran' in spec))

        if "+mpi" in spec:
            cfg.write(cmake_cache_option("ENABLE_MPI", '+mpi' in spec))
            cfg.write(cmake_cache_entry("MPI_CXX_COMPILER", spec['mpi'].mpicxx))

        cfg.write(cmake_cache_option("ENABLE_NUMA", '+numa' in spec))
        cfg.write(cmake_cache_option("ENABLE_OPENMP", '+openmp' in spec))
        if "+openmp_target" in spec:
            cfg.write(cmake_cache_option("ENABLE_OPENMP_TARGET", True))
            if ('%xl' in spec):
                cfg.write(cmake_cache_entry("OpenMP_CXX_FLAGS", "-qsmp;-qoffload"))

        cfg.write(cmake_cache_option("ENABLE_BENCHMARKS", 'tests=benchmarks' in spec))
        cfg.write(cmake_cache_option("ENABLE_TESTS", not 'tests=none' in spec))
        cfg.write(cmake_cache_option("ENABLE_TOOLS", '+tools' in spec))
        cfg.write(cmake_cache_option("ENABLE_WARNINGS_AS_ERRORS", '+werror' in spec))
        cfg.write(cmake_cache_option("ENABLE_SANITIZER_TESTS", '+sanitizer_tests' in spec))

        #######################
        # Close and save
        #######################
        cfg.write("\n")
        cfg.close()

        print("OUT: host-config file {0}".format(host_config_path))

    def cmake_args(self):
        spec = self.spec
        host_config_path = self._get_host_config_path(spec)

        options = []
        options.extend(['-C', host_config_path])

        return options

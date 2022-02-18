# Copyright 2013-2022 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


from spack import *

import socket
import os

from os import environ as env
from os.path import join as pjoin

import re


class Umpire(CachedCMakePackage, CudaPackage, ROCmPackage):
    """An application-focused API for memory management on NUMA & GPU
    architectures"""

    homepage = 'https://github.com/LLNL/Umpire'
    git      = 'https://github.com/LLNL/Umpire.git'

    version('develop', branch='develop', submodules='True')
    version('main', branch='main', submodules='True')
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
    variant('ipc_shmem', default=False, description='Enable POSIX shared memory')
    variant('sqlite', default=False, description='Enable sqlite integration with umpire events')
    variant('numa', default=False, description='Enable NUMA support')
    variant('shared', default=False, description='Enable Shared libs')
    variant('openmp', default=False, description='Build with OpenMP support')
    variant('openmp_target', default=False, description='Build with OpenMP 4.5 support')
    variant('deviceconst', default=False,
            description='Enables support for constant device memory')
    variant('tests', default='basic', values=('none', 'basic', 'benchmarks'),
            multi=False, description='Tests to run')

    variant('libcpp', default=False, description='Uses libc++ instead of libstdc++')
    variant('tools', default=True, description='Enable tools')
    variant('dev_benchmarks', default=False, description='Enable Developer Benchmarks')
    variant('werror', default=True, description='Enable warnings as errors')
    variant('asan', default=False, description='Enable ASAN')
    variant('sanitizer_tests', default=False, description='Enable address sanitizer tests')

    depends_on('cmake@3.14:', type='build')
    depends_on('sqlite', when='+sqlite')
    depends_on('mpi', when='+mpi')

    depends_on('blt@0.4.1', type='build', when='@main')
    depends_on('blt@develop', type='build')

    # variants +rocm and amdgpu_targets are not automatically passed to
    # dependencies, so do it manually.
    depends_on('camp+rocm', when='+rocm')
    for val in ROCmPackage.amdgpu_targets:
        depends_on('camp amdgpu_target=%s' % val, when='amdgpu_target=%s' % val)

    depends_on('camp+cuda', when='+cuda')
    for sm_ in CudaPackage.cuda_arch_values:
        depends_on('camp cuda_arch={0}'.format(sm_),
                   when='cuda_arch={0}'.format(sm_))

    depends_on('camp@0.1.0', when='@main')
    depends_on('camp@0.2.2')

    conflicts('+numa', when='@:0.3.2')
    conflicts('~c', when='+fortran', msg='Fortran API requires C API')
    conflicts('~openmp', when='+openmp_target', msg='OpenMP target requires OpenMP')
    conflicts('+cuda', when='+rocm')
    conflicts('+openmp', when='+rocm')
    conflicts('+openmp_target', when='+rocm')
    conflicts('+deviceconst', when='~rocm~cuda')
    conflicts('~mpi', when='+ipc_shmem', msg='Shared Memory Allocator requires MPI')
    conflicts('+ipc_shmem', when='@:5.0.1')
    conflicts('+sqlite', when='@:6.0.0')
    conflicts('+sanitizer_tests', when='~asan')

    def _get_sys_type(self, spec):
        sys_type = str(spec.architecture)
        # if on llnl systems, we can use the SYS_TYPE
        if "SYS_TYPE" in env:
            sys_type = env["SYS_TYPE"]
        return sys_type

    @property
    def cache_name(self):
        hostname = socket.gethostname()
        if "SYS_TYPE" in env:
            hostname = hostname.rstrip('1234567890')
        return "{0}-{1}-{2}@{3}.cmake".format(
            hostname,
            self._get_sys_type(self.spec),
            self.spec.compiler.name,
            self.spec.compiler.version
        )

    def spec_uses_toolchain(self, spec):
        gcc_toolchain_regex = re.compile(".*gcc-toolchain.*")
        using_toolchain = list(filter(gcc_toolchain_regex.match, spec.compiler_flags['cxxflags']))

        return using_toolchain

    def spec_uses_gccname(self, spec):
        gcc_name_regex = re.compile(".*gcc-name.*")
        using_gcc_name = list(filter(gcc_name_regex.match, spec.compiler_flags['cxxflags']))

        return using_gcc_name



    def initconfig_compiler_entries(self):
        spec = self.spec
        entries = super(Umpire, self).initconfig_compiler_entries()

        entries.append(cmake_cache_option("ENABLE_FORTRAN", 
            ('+fortran' in spec) and (self.compiler.fc is not None)))
        entries.append(cmake_cache_option("UMPIRE_ENABLE_C", '+c' in spec))
        
        fortran_compilers = ["gfortran", "xlf"]
        if any(compiler in self.compiler.fc for compiler in fortran_compilers) and ("clang" in self.compiler.cxx):
            entries.append(cmake_cache_string("BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE",
            "/usr/tce/packages/gcc/gcc-4.9.3/lib64;/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3;/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64;/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/x86_64-unknown-linux-gnu/4.9.3"))

            libdir = pjoin(os.path.dirname(
                           os.path.dirname(self.compiler.fc)), "lib")
            flags = ""
            for _libpath in [libdir, libdir + "64"]:
                if os.path.exists(_libpath):
                    flags += " -Wl,-rpath,{0}".format(_libpath)
            description = ("Adds a missing libstdc++ rpath")
            if flags:
                entries.append(cmake_cache_string("BLT_EXE_LINKER_FLAGS", flags, description))


        compilers_using_toolchain = ["pgi", "xl", "icpc"]
        if any(compiler in self.compiler.cxx for compiler in compilers_using_toolchain):
            if self.spec_uses_toolchain(self.spec) or self.spec_uses_gccname(self.spec):
                entries.append(cmake_cache_string("BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE",
                "/usr/tce/packages/gcc/gcc-4.9.3/lib64;/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3;/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64;/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/x86_64-unknown-linux-gnu/4.9.3"))

        entries = [x for x in entries if not 'COMPILER_ID' in x]
        return entries


    def initconfig_hardware_entries(self):
        spec = self.spec
        entries = super(Umpire, self).initconfig_hardware_entries()

        entries.append(cmake_cache_option("ENABLE_CUDA", "+cuda" in spec))
        if "+cuda" in spec:
            cuda_flags = []
            if not spec.satisfies('cuda_arch=none'):
                cuda_arch = spec.variants['cuda_arch'].value
                cuda_flags.append('-arch sm_{0}'.format(cuda_arch[0]))

            if self.spec_uses_toolchain(self.spec):
                cuda_flags.append("-Xcompiler {}".format(self.spec_uses_toolchain(self.spec)[0]))

            if (spec.satisfies('%gcc@8.1: target=ppc64le')):
                cuda_flags.append('-Xcompiler -mno-float128')

            entries.append(cmake_cache_string("CMAKE_CUDA_FLAGS",  ' '.join(cuda_flags)))

        entries.append(cmake_cache_option("ENABLE_HIP", "+rocm" in spec))
        if "+rocm" in spec:
            hip_root = spec['hip'].prefix
            rocm_root = hip_root + "/.."
            entries.append(cmake_cache_path("HIP_ROOT_DIR",
                                        hip_root))
            entries.append(cmake_cache_path("HIP_CLANG_PATH",
                                        rocm_root + '/llvm/bin'))
            entries.append(cmake_cache_string("HIP_HIPCC_FLAGS",
                                        '--amdgpu-target=gfx906'))
            entries.append(cmake_cache_string("HIP_RUNTIME_INCLUDE_DIRS",
                                        "{0}/include;{0}/../hsa/include".format(hip_root)))
            hip_link_flags = "-Wl,--disable-new-dtags -L{0}/lib -L{0}/../lib64 -L{0}/../lib -Wl,-rpath,{0}/lib:{0}/../lib:{0}/../lib64 -lamdhip64 -lhsakmt -lhsa-runtime64".format(hip_root)
            if '%gcc' in spec:
                gcc_bin = os.path.dirname(self.compiler.cxx)
                gcc_prefix = join_path(gcc_bin, '..')
                entries.append(cmake_cache_string("HIP_CLANG_FLAGS", "--gcc-toolchain={0}".format(gcc_prefix))) 
                entries.append(cmake_cache_string("CMAKE_EXE_LINKER_FLAGS", hip_link_flags + " -Wl,-rpath {}/lib64".format(gcc_prefix)))
            else:
                entries.append(cmake_cache_string("CMAKE_EXE_LINKER_FLAGS", hip_link_flags))

        entries.append(cmake_cache_option("UMPIRE_ENABLE_DEVICE_CONST", "+deviceconst" in spec))

        entries.append(cmake_cache_option("UMPIRE_ENABLE_OPENMP_TARGET", "+openmp_target" in spec))
        if "+openmp_target" in spec:
            if ('%xl' in spec):
                entries.append(cmake_cache_string("OpenMP_CXX_FLAGS", "-qsmp;-qoffload"))

        return entries


    def initconfig_mpi_entries(self):
        spec = self.spec

        entries = super(Umpire, self).initconfig_mpi_entries()
        entries.append(cmake_cache_option("ENABLE_MPI", '+mpi' in spec))

        return entries


    def initconfig_package_entries(self):
        spec = self.spec
        entries = []

        entries.append(cmake_cache_path("BLT_SOURCE_DIR", spec['blt'].prefix))
        entries.append(cmake_cache_path("camp_DIR" ,spec['camp'].prefix))
        entries.append(cmake_cache_string("CMAKE_BUILD_TYPE", spec.variants['build_type'].value))
        entries.append(cmake_cache_option("ENABLE_BENCHMARKS", 'tests=benchmarks' in spec or '+dev_benchmarks' in spec))
        entries.append(cmake_cache_option("UMPIRE_ENABLE_DEVELOPER_BENCHMARKS", '+dev_benchmarks' in spec))
        entries.append(cmake_cache_option("ENABLE_TESTS", not 'tests=none' in spec))
        entries.append(cmake_cache_option("UMPIRE_ENABLE_TOOLS", '+tools' in spec))
        entries.append(cmake_cache_option("ENABLE_WARNINGS_AS_ERRORS", '+werror' in spec))
        entries.append(cmake_cache_option("UMPIRE_ENABLE_ASAN", '+asan' in spec))
        entries.append(cmake_cache_option("UMPIRE_ENABLE_SANITIZER_TESTS", '+sanitizer_tests' in spec))
        entries.append(cmake_cache_option("ENABLE_NUMA", '+numa' in spec))
        entries.append(cmake_cache_option("ENABLE_OPENMP", '+openmp' in spec))
        entries.append(cmake_cache_option("UMPIRE_ENABLE_IPC_SHARED_MEMORY", '+ipc_shmem' in spec))
        entries.append(cmake_cache_option("UMPIRE_ENABLE_SQLITE", '+sqlite' in spec))
        
        return entries


    def cmake_args(self):
        spec = self.spec
        options = []
        return options

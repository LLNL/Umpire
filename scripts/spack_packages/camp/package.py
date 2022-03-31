# Copyright 2013-2022 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class Camp(CMakePackage, CudaPackage, ROCmPackage):
    """
    Compiler agnostic metaprogramming library providing concepts,
    type operations and tuples for C++ and cuda
    """

    homepage = "https://github.com/LLNL/camp"
    git      = "https://github.com/LLNL/camp.git"
    url      = "https://github.com/LLNL/camp/archive/refs/tags/v2022.03.0.tar.gz"

    maintainers = ['trws']

    version('main', branch='main', submodules='True')
    version('2022.03.0', sha256='e9090d5ee191ea3a8e36b47a8fe78f3ac95d51804f1d986d931e85b8f8dad721')
    version('0.3.0', sha256='129431a049ca5825443038ad5a37a86ba6d09b2618d5fe65d35f83136575afdb')
    version('0.2.2', sha256='194d38b57e50e3494482a7f94940b27f37a2bee8291f2574d64db342b981d819')
    version('0.1.0', sha256='fd4f0f2a60b82a12a1d9f943f8893dc6fe770db493f8fae5ef6f7d0c439bebcc')

    variant('tests', default=False, description='Build tests')

    def cmake_args(self):
        spec = self.spec

        options = []

        options.append(self.define_from_variant('ENABLE_TESTS', 'tests'))

        return 

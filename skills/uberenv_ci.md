# Reproducing CI with Uberenv

Uberenv automates building Umpire with Spack to reproduce CI pipelines locally.

## Basic Usage

```bash
# Build with default spec
python3 scripts/uberenv/uberenv.py

# Build with specific spec
python3 scripts/uberenv/uberenv.py --spec="+cuda %clang@14"

# Install Umpire (not just dependencies)
python3 scripts/uberenv/uberenv.py --install

# Specify install prefix
python3 scripts/uberenv/uberenv.py --prefix=/path/to/install
```

## Configuration

Uberenv reads `.uberenv_config.json` for defaults:
- package_name: "umpire"
- spack_configs_path: "scripts/radiuss-spack-configs"
- spack_packages_path: Contains Umpire spack package definition

## Reproducing CI Jobs

CI job specs are defined in `.gitlab/jobs/<machine>.yml`:
- lassen.yml: IBM Power9 + CUDA
- tioga.yml: AMD MI300A + ROCm
- tuolumne.yml: AMD MI300A + ROCm
- corona.yml: AMD MI60 + ROCm
- dane.yml: Intel systems

Example reproducing a CI job:
```bash
# From lassen.yml: clang_14_0_5_gcc_8_3_1_cuda_11_7_0
python3 scripts/uberenv/uberenv.py \
  --spec="+cuda %clang-14 ^cuda@11.7.0+allow-unsupported-compilers"

# From tioga.yml: cce_15_0_1_hip_5_7_1
python3 scripts/uberenv/uberenv.py \
  --spec="+rocm %cce@15.0.1 ^hip@5.7.1"
```

## Common Specs

Host-only builds:
```bash
python3 scripts/uberenv/uberenv.py --spec="~cuda ~rocm %gcc@11"
```

CUDA builds:
```bash
python3 scripts/uberenv/uberenv.py --spec="+cuda %clang@14"
```

ROCm/HIP builds:
```bash
python3 scripts/uberenv/uberenv.py --spec="+rocm %clang@16"
```

With additional features:
```bash
# With OpenMP
python3 scripts/uberenv/uberenv.py --spec="+openmp %gcc"

# With MPI and shared memory
python3 scripts/uberenv/uberenv.py --spec="+mpi +ipc_shmem %clang"

# With tests
python3 scripts/uberenv/uberenv.py --spec="tests=basic %gcc"
```

## Useful Options

```bash
# Only setup spack, don't build
python3 scripts/uberenv/uberenv.py --setup-only

# Setup and generate environment script
python3 scripts/uberenv/uberenv.py --setup-and-env-only

# Use existing spack instance
python3 scripts/uberenv/uberenv.py --upstream=/path/to/spack

# Clean previous build
python3 scripts/uberenv/uberenv.py --clean

# Run tests during build
python3 scripts/uberenv/uberenv.py --run_tests

# Parallel build jobs
python3 scripts/uberenv/uberenv.py -j 16
```

## Spack Build Modes

```bash
# dev-build (default): Uses spack dev-build
python3 scripts/uberenv/uberenv.py --spack-build-mode=dev-build

# install: Uses spack install
python3 scripts/uberenv/uberenv.py --spack-build-mode=install
```

## Generated Files

After running uberenv:
- `uberenv_libs/`: Spack installation directory
- `<hostname>-<spec-hash>.cmake`: Host config file
- Use host config with CMake: `cmake -C <host-config>.cmake ..`

## Debugging

```bash
# Enable spack debug output
python3 scripts/uberenv/uberenv.py --spack-debug

# See what spack would do
python3 scripts/uberenv/uberenv.py --setup-only

# Check generated spack environment
cat uberenv_libs/spack/var/spack/environments/umpire/spack.yaml
```

## Common Variants

Umpire spack variants:
- +cuda / ~cuda: CUDA support
- +rocm / ~rocm: ROCm/HIP support
- +openmp / ~openmp: OpenMP support
- +mpi / ~mpi: MPI support
- +fortran / ~fortran: Fortran bindings
- +ipc_shmem: IPC shared memory
- +mpi3_shmem: MPI-3 shared memory
- +tools: Build tools
- +examples: Build examples
- tests=basic/none: Build tests

## Machine-Specific Jobs

To see available jobs for a specific machine:
- Check `.gitlab/jobs/<machine>.yml`
- Look for SPEC: variables
- Reproduce with: `--spec="<spec_content>"`

Resources:
- Uberenv docs: https://uberenv.readthedocs.io/
- Umpire uberenv guide: https://umpire.readthedocs.io/en/develop/sphinx/developer/uberenv.html
- RADIUSS Spack configs: scripts/radiuss-spack-configs/

set (allocation_map_tests_depends
  umpire umpire_detail gtest)

if (ENABLE_CUDA)
  set (allocation_map_tests_depends
    ${allocation_map_tests_depends}
    cuda_runtime)
endif ()
if (ENABLE_HIP)
  set (allocation_map_tests_depends
    ${allocation_map_tests_depends}
    hip_runtime)
endif ()
if (ENABLE_OPENMP_TARGET)
  set (allocation_map_tests_depends
    ${allocation_map_tests_depends}
    openmp)
endif ()

blt_add_executable(
  NAME allocation_map_tests
  SOURCES allocation_map_tests.cpp
  DEPENDS_ON ${allocation_map_tests_depends})

blt_add_test(
  NAME allocation_map_tests
  COMMAND allocation_map_tests)

blt_add_executable(
  NAME memory_map_tests
  SOURCES memory_map_tests.cpp
  DEPENDS_ON umpire umpire_detail gtest)

blt_add_test(
  NAME memory_map_tests
  COMMAND memory_map_tests)

blt_add_executable(
  NAME output_buffer_tests
  SOURCES output_buffer_tests.cpp
  DEPENDS_ON umpire umpire_detail gtest)

blt_add_test(
  NAME output_buffer_tests
  COMMAND output_buffer_tests)
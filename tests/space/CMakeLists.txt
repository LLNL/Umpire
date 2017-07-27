blt_add_executable(
  NAME space_registry_tests
  SOURCES space_registry_tests.cpp
  DEPENDS_ON umpire_space umpire_util umpire gtest)

blt_add_test(
  NAME space_registry_tests
  COMMAND space_registry_tests)

blt_add_executable(
  NAME cnmem_pool_tests
  SOURCES cnmem_pool_test.cpp
  DEPENDS_ON umpire_space umpire_tpl_cnmem cuda_runtime gtest)

blt_add_test(
  NAME cnmem_pool_tests
  COMMAND cnmem_pool_tests)

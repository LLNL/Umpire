blt_add_executable(
  NAME space_registry_tests
  SOURCES space_registry_tests.cpp
  DEPENDS_ON umpire_space umpire_util gtest)

blt_add_test(
  NAME space_registry_tests
  COMMAND space_registry_tests)

blt_add_executable(
  NAME resource_manager_tests
  SOURCES resource_manager_tests.cpp
  DEPENDS_ON umpire gtest)

blt_add_executable(
  NAME stdlib_alloc_tests
  SOURCES stdlib_alloc_tests.cpp
  DEPENDS_ON umpire gtest)

blt_add_test(
  NAME stdlib_alloc_test
  COMMAND stdlib_alloc_tests)

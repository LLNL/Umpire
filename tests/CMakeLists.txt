blt_add_executable(
  NAME resource_manager_tests
  SOURCES resource_manager_tests.cpp
  DEPENDS_ON umpire gtest)

blt_add_test(
  NAME resource_manager_tests
  COMMAND resource_manager_tests)

blt_add_executable(
  NAME stdlib_alloc_tests
  SOURCES stdlib_alloc_tests.cpp
  DEPENDS_ON umpire gtest)

blt_add_test(
  NAME stdlib_alloc_test
  COMMAND stdlib_alloc_tests)

add_subdirectory(alloc)
add_subdirectory(space)

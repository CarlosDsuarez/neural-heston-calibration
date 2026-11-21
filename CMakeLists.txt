cmake_minimum_required(VERSION 3.26)
project(neural_heston_calibration VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(NHC_BUILD_TESTS "Build C++ Catch2 unit tests" ON)
option(NHC_BUILD_PYTHON "Build nanobind Python module nhc_core" OFF)
option(NHC_ENABLE_OPENMP "Enable OpenMP for parallel synthetic dataset generation" ON)

include(FetchContent)

# ---------------------------------------------------------------------------
# Eigen 5.0.1 (pinned, header-only)
# ---------------------------------------------------------------------------
set(BUILD_TESTING_SAVED ${BUILD_TESTING})
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_BLAS OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_LAPACK OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(eigen)

set(BUILD_TESTING ${BUILD_TESTING_SAVED} CACHE BOOL "" FORCE)

# ---------------------------------------------------------------------------
# OpenMP (optional, supports macOS Homebrew libomp)
# ---------------------------------------------------------------------------
if(NHC_ENABLE_OPENMP)
  if(APPLE AND EXISTS "/opt/homebrew/opt/libomp")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
  endif()
  find_package(OpenMP)
endif()

# ---------------------------------------------------------------------------
# Core header-only C++20 numerical library
# Note (Spec T0.2): -O3 and -fno-math-errno in Release; NO -ffast-math on
# cos_pricer / heston_cf because complex branch cuts require strict IEEE-754.
# ---------------------------------------------------------------------------
add_library(nhc_headers INTERFACE)
target_include_directories(nhc_headers INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>
)
target_link_libraries(nhc_headers INTERFACE Eigen3::Eigen)
target_compile_features(nhc_headers INTERFACE cxx_std_20)

target_compile_definitions(nhc_headers INTERFACE EIGEN_DONT_PARALLELIZE)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(nhc_headers INTERFACE
    $<$<CONFIG:Release>:-O3 -fno-math-errno>
    $<$<CONFIG:RelWithDebInfo>:-O3 -fno-math-errno>
    -Wall -Wextra -Wpedantic
  )
endif()

# OpenMP is linked selectively to standalone parallel C++ tools (e.g., Phase 3 dataset generator)
# rather than nhc_core, avoiding duplicate libomp conflicts when PyTorch is imported in Python.
add_library(nhc_openmp INTERFACE)
if(NHC_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(nhc_openmp INTERFACE OpenMP::OpenMP_CXX)
  target_compile_definitions(nhc_openmp INTERFACE NHC_HAS_OPENMP=1)
endif()

# ---------------------------------------------------------------------------
# Parallel OpenMP Dataset Generator CLI (Phase 3 T3.3)
# ---------------------------------------------------------------------------
option(NHC_BUILD_TOOLS "Build standalone C++ CLI tools (nhc_gen_dataset)" ON)
if(NHC_BUILD_TOOLS)
  add_executable(nhc_gen_dataset cpp/src/gen_dataset.cpp)
  target_link_libraries(nhc_gen_dataset PRIVATE nhc_headers nhc_openmp)
endif()

# ---------------------------------------------------------------------------
# C++ Unit Tests (Catch2 v3.9.1)
# ---------------------------------------------------------------------------
if(NHC_BUILD_TESTS)
  enable_testing()
  FetchContent_Declare(
    Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.9.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(Catch2)
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(CTest)
  include(Catch)

  add_executable(nhc_tests
    cpp/tests/test_smoke.cpp
    cpp/tests/test_cos_pricer.cpp
    cpp/tests/test_calibration.cpp
    cpp/tests/test_grid.cpp
  )
  target_link_libraries(nhc_tests PRIVATE nhc_headers Catch2::Catch2WithMain)
  catch_discover_tests(nhc_tests)
endif()

# ---------------------------------------------------------------------------
# Python Bindings (nanobind)
# ---------------------------------------------------------------------------
if(NHC_BUILD_PYTHON)
  find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
  )
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(nhc_core STABLE_ABI NB_STATIC
    cpp/src/bindings.cpp
  )
  target_link_libraries(nhc_core PRIVATE nhc_headers)

  install(TARGETS nhc_core LIBRARY DESTINATION nhc)
endif()

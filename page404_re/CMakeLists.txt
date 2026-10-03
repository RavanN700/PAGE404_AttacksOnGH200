cmake_minimum_required(VERSION 3.18)

# CUDA project for the eviction_set_builder tool.
# Target hardware: NVIDIA GH200 (Hopper, compute capability 9.0 -> sm_90).
project(page404_re LANGUAGES CXX CUDA)

# --- Standards --------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- Target architecture ----------------------------------------------------
# GH200 is Hopper (sm_90). Override on the command line with:
#   cmake -DCMAKE_CUDA_ARCHITECTURES=90 ..
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

# --- Host compiler ----------------------------------------------------------
# On this machine the default 'gcc' is v12 but only gcc-11 ships a working
# cc1plus, so point nvcc at g++-11. Override with -DCMAKE_CUDA_HOST_COMPILER=.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  find_program(_GXX11 g++-11)
  if(_GXX11)
    set(CMAKE_CUDA_HOST_COMPILER "${_GXX11}")
  endif()
endif()

# --- Build type -------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- Executables ------------------------------------------------------------
# One standalone tool per .cu source. They share no translation units, so each
# is built independently. The sources use getopt / mmap (POSIX) and the CUDA
# runtime, which the CUDA language target links automatically -- no extra
# libraries are required.
set(TOOLS
  eviction_set_builder
  reaction_to_counter_values
  produce_unique_sets
  collect_set_of_addresses
)

foreach(tool ${TOOLS})
  add_executable(${tool} ${tool}.cu)
  # Host-side optimization flags passed through to the host compiler by nvcc.
  target_compile_options(${tool} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  )
endforeach()

# The engine: C++20 + CUDA for GB10 (sm_121a) only.
# Each line of work owns one subdirectory with its own CMakeLists.txt; it is picked up when present.
cmake_minimum_required(VERSION 3.28)
if(NOT DEFINED CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.0/bin/nvcc)
endif()
# Block-scaled MMA needs the arch-specific target: compute_121a/sm_121a (see docs/findings.md#tensor-cores).
set(CMAKE_CUDA_ARCHITECTURES OFF)
project(dsgb10 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The tensor-parallel world this build serves: one constant (common/world.h) that every per-rank shape
# derives from. 4 is the fleet's; 2 and 1 divide the checkpoint's shapes too (3 divides none of them).
# One build per world: $TP must agree with it at startup (common/tp.h), and weights, Engram shards and
# fleet files cut for another world are refused.
set(DSGB10_WORLD 4 CACHE STRING "Tensor-parallel world this build serves: 1, 2 or 4")
if(NOT DSGB10_WORLD MATCHES "^[124]$")
  message(FATAL_ERROR "DSGB10_WORLD=${DSGB10_WORLD}: the engine builds for world 1, 2 or 4 "
                      "(3 does not divide the checkpoint's 129280 vocabulary rows, 5120 hidden, 64 heads "
                      "or 1280 q-lora rows)")
endif()
add_compile_definitions(DSGB10_WORLD=${DSGB10_WORLD})
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_121a,code=sm_121a --expt-relaxed-constexpr -lineinfo")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

foreach(dir common weights comm kernels attn dspark engram model server tests)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt)
    add_subdirectory(${dir})
  endif()
endforeach()

# This project's fleet file beside the binaries that read one (dsgb10_rank, ring_test, coll_tests), as
# fleet.conf, where they look by default; run4.sh and service.sh install their FLEET beside the copies
# they run, and --fleet FILE reads another.
foreach(dir server comm model)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../fleet/homelab.conf ${CMAKE_BINARY_DIR}/${dir}/fleet.conf COPYONLY)
endforeach()

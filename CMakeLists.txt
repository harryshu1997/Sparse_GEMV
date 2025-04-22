cmake_minimum_required(VERSION 3.10)
project(SparseInnerProduct LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ——— Android settings (if building for Android) ———
if(ANDROID)
  set(CMAKE_ANDROID_ARCH_ABI       "arm64-v8a")
  set(CMAKE_ANDROID_NDK            "/home/monsterharry/Android/Sdk/ndk/android-ndk-r25c/")
  set(CMAKE_ANDROID_STL_TYPE       "c++_shared")
  # The toolchain file is normally passed via -DCMAKE_TOOLCHAIN_FILE
endif()

# ——— Manually point to your cross‑compiled OpenBLAS ———
set(OpenBLAS_ROOT        "/home/monsterharry/Documents/OpenBLAS/install-android-arm64v8a")
set(OpenBLAS_INCLUDE_DIR "${OpenBLAS_ROOT}/include")
set(OpenBLAS_LIB         "${OpenBLAS_ROOT}/lib/libopenblas.so")

# ——— Find OpenCL library ———
find_library(OPENCL_LIB_PATH
  NAMES OpenCL opencl
  PATHS ${CMAKE_SOURCE_DIR}/lib /usr/lib /usr/local/lib
)
if(NOT OPENCL_LIB_PATH)
  message(FATAL_ERROR "Could not find OpenCL library. Set OPENCL_LIB_PATH manually.")
endif()

# ——— Source file lists ———
set(CPU_SOURCES
    src/main.cpp
    src/cpu/inner_product_cpu.cpp
    src/cpu/dense_gemv_cpu.cpp
)

set(GPU_SOURCES
    src/main.cpp
    src/cpu/inner_product_cpu.cpp
    src/gpu/inner_product_opencl.cpp
    src/cpu/dense_gemv_cpu.cpp
    src/gpu/dense_gemv_opencl.cpp
)

# ——— Global include directories ———
include_directories(
  ${CMAKE_SOURCE_DIR}/src/types
  ${CMAKE_SOURCE_DIR}/src/cpu
  ${CMAKE_SOURCE_DIR}/src/gpu
)

# ——— CPU executable (BLAS only) ———
add_executable(inner_product_cpu ${CPU_SOURCES})
target_include_directories(inner_product_cpu PRIVATE ${OpenBLAS_INCLUDE_DIR})
target_link_libraries(inner_product_cpu PRIVATE ${OpenBLAS_LIB})

# ——— GPU executable (OpenCL + BLAS) ———
option(ENABLE_OPENCL "Build the OpenCL (GPU) version" ON)
add_executable(inner_product_gpu ${GPU_SOURCES})

# Enable the OpenCL code path
target_compile_definitions(inner_product_gpu PRIVATE ENABLE_OPENCL)

# Include OpenBLAS headers
target_include_directories(inner_product_gpu PRIVATE ${OpenBLAS_INCLUDE_DIR})

# (Optional) If you have OpenCL headers in a custom location:
if(DEFINED OPENCL_INCLUDE_DIRS)
  target_include_directories(inner_product_gpu PRIVATE ${OPENCL_INCLUDE_DIRS})
endif()

# Link against OpenCL and OpenBLAS
target_link_libraries(inner_product_gpu
  PRIVATE
    ${OPENCL_LIB_PATH}
    ${OpenBLAS_LIB}
)
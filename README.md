# Sparse Inner Product Project

## Project Overview

This project implements a sparse inner product calculation using both CPU and OpenCL GPU versions. It is designed to efficiently handle sparse data structures and perform computations on them.

## Project Structure

- `CMakeLists.txt`: The build configuration file that defines compilation options, source files, and target platform settings.

- `README.md`: This documentation file provides information on how to use and build the project.

- `src/cpu/inner_product_cpu.cpp`: Implements the CPU version of the inner product calculation. It defines the main computation function and handles the storage and access of sparse data.

- `src/gpu/inner_product_opencl.cpp`: Implements the OpenCL GPU version of the inner product calculation. It sets up the OpenCL environment, loads the kernel, and invokes the kernel for computation.

- `src/gpu/inner_product_kernel.cl`: Contains the OpenCL kernel code that defines how to perform the sparse inner product calculation on the GPU.

- `src/types/sparse_types.h`: Defines the data structures and related types for sparse vectors and matrices, providing the basic types used in both CPU and GPU versions.

- `3rd_party`: Contains third-party libraries that may be used for various functionalities, including:
  - `cutlass`: Source code for the Cutlass library, potentially for high-performance CUDA computations.
  - `flatbuffers`: Source code for the FlatBuffers library, potentially for efficient data serialization.
  - `half`: Implementation of Half data types, potentially for half-precision floating-point computations.
  - `imageHelper`: Auxiliary tools for image processing.
  - `lic`: License information.
  - `OpenCLHeaders`: OpenCL header files providing interfaces for the OpenCL API.
  - `protobuf`: Source code for the Protocol Buffers library, potentially for data serialization.
  - `rapidjson`: Source code for the RapidJSON library, potentially for efficient JSON parsing.

- `android/toolchain.cmake`: The CMake configuration file for the Android toolchain, defining the settings required for cross-compilation.

- `android/README.md`: Documentation regarding the Android build, providing information on how to build the project for Android.
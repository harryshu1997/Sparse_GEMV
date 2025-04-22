# README for Android Build

This project provides a cross-platform implementation of sparse inner product calculations, with support for both CPU and OpenCL GPU versions. This README outlines the steps necessary to configure and build the project for Android.

## Project Structure

- **CMakeLists.txt**: The main build configuration file that defines compilation options, source files, and target platform settings.
- **src/cpu/inner_product_cpu.cpp**: Implements the CPU version of the inner product calculation, handling the storage and access of sparse data.
- **src/gpu/inner_product_opencl.cpp**: Implements the OpenCL GPU version of the inner product calculation, setting up the OpenCL environment, loading the kernel, and invoking it for computation.
- **src/gpu/inner_product_kernel.cl**: Contains the OpenCL kernel code that defines how to perform the sparse inner product calculation on the GPU.
- **src/types/sparse_types.h**: Defines the data structures and related types for sparse vectors and matrices.

## Building for Android

To build the project for Android, follow these steps:

1. **Install Android NDK**: Ensure that you have the Android NDK installed on your system. You can download it from the [Android NDK website](https://developer.android.com/ndk).

2. **Set Up the Toolchain**: The `toolchain.cmake` file in the `android` directory is configured for cross-compilation. Make sure to specify the path to your Android NDK in this file.

3. **Configure the Build**:
   - Open a terminal and navigate to the root of the project directory.
   - Create a build directory for the Android build:
     ```
     mkdir build_android
     cd build_android
     ```
   - Run CMake with the Android toolchain:
     ```
     cmake -DCMAKE_TOOLCHAIN_FILE=../android/toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DENABLE_OPENCL=ON -DOPENCL_LIB_PATH=../libOpenCL.so ..
     ```

4. **Build the Project**:
   - After configuring, build the project using:
     ```
     cmake --build .
     ```

5. **Run the Application**: Once the build is complete, you can deploy the generated APK to your Android device using `adb` or any other deployment method.

## Additional Information

For more details on how to use and modify the project, refer to the main `README.md` file located in the root of the project directory.
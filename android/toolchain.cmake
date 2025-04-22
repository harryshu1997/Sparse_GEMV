set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_VERSION 28)  # Minimum Android API level
set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)  # Target architecture
set(CMAKE_ANDROID_NDK /home/monsterharry/Android/Sdk/ndk/android-ndk-r25c)  # Path to your Android NDK
set(CMAKE_ANDROID_STL_TYPE c++_shared)  # Use shared C++ standard library

# # Specify the toolchain file for cross-compilation
# set(CMAKE_C_COMPILER ${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang)
# set(CMAKE_CXX_COMPILER ${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++)
# set(CMAKE_C_COMPILER_TARGET aarch64-linux-android)
# set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-android)

# Set the path for the OpenCL headers and libraries
set(OPENCL_INCLUDE_DIRS /home/monsterharry/Documents/sparse_gemv/3rd_party/OpenCLHeaders)
set(OPENCL_LIBRARIES OpenCL)  

# Add the OpenCL include directories
# include_directories(${OPENCL_INCLUDE_DIRS})

# Link OpenCL library
find_library(OPENCL_LIB NAMES OpenCL
    PATHS
        # ${CMAKE_ANDROID_NDK}/sources/CL/lib
        /system/vendor/lib
        /system/vendor/lib64
        /system/lib
        /system/lib64
    NO_DEFAULT_PATH
)  # Ensure OpenCL library is found

# target_link_libraries(inner_product_gpu ${OPENCL_LIB})
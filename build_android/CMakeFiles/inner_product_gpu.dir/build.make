# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/monsterharry/Documents/sparse_gemv

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/monsterharry/Documents/sparse_gemv/build_android

# Include any dependencies generated for this target.
include CMakeFiles/inner_product_gpu.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/inner_product_gpu.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/inner_product_gpu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/inner_product_gpu.dir/flags.make

CMakeFiles/inner_product_gpu.dir/src/main.cpp.o: CMakeFiles/inner_product_gpu.dir/flags.make
CMakeFiles/inner_product_gpu.dir/src/main.cpp.o: /home/monsterharry/Documents/sparse_gemv/src/main.cpp
CMakeFiles/inner_product_gpu.dir/src/main.cpp.o: CMakeFiles/inner_product_gpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/monsterharry/Documents/sparse_gemv/build_android/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/inner_product_gpu.dir/src/main.cpp.o"
	/home/monsterharry/Android/Sdk/ndk/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android28 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/inner_product_gpu.dir/src/main.cpp.o -MF CMakeFiles/inner_product_gpu.dir/src/main.cpp.o.d -o CMakeFiles/inner_product_gpu.dir/src/main.cpp.o -c /home/monsterharry/Documents/sparse_gemv/src/main.cpp

CMakeFiles/inner_product_gpu.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/inner_product_gpu.dir/src/main.cpp.i"
	/home/monsterharry/Android/Sdk/ndk/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android28 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/monsterharry/Documents/sparse_gemv/src/main.cpp > CMakeFiles/inner_product_gpu.dir/src/main.cpp.i

CMakeFiles/inner_product_gpu.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/inner_product_gpu.dir/src/main.cpp.s"
	/home/monsterharry/Android/Sdk/ndk/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android28 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/monsterharry/Documents/sparse_gemv/src/main.cpp -o CMakeFiles/inner_product_gpu.dir/src/main.cpp.s

CMakeFiles/inner_product_gpu.dir/src/cpu/inner_product_cpu.cpp.o: CMakeFiles/inner_product_gpu.dir/flags.make
CMakeFiles/inner_product_gpu.dir/src/cpu/inner_product_cpu.cpp.o: /home/monsterharry/Documents/sparse_gemv/src/cpu/inner_product_cpu.cpp
CMakeFiles/inner_product_gpu.dir/src/cpu/inner_product_cpu.cpp.o: CMakeFiles/inner_product_gpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/monsterharry/Documents/sparse_gemv/build_android/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/inner_product_gpu.dir/src/cpu/inner_product_cpu.cpp.o"
	/home/monsterharry/Android/Sdk/ndk/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android28 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/inner_product_gpu.dir/src/cpu/inner_product_cpu.cpp.o -MF CMakeFiles/inner_product_gpu.dir/src/cpu/inner_product_cpu.cpp.o.d -o CMakeFiles/inner_product_gpu.dir/src/cpu/inner_product_cpu.cpp.o -c /home/monsterharry/Documents/sparse_gemv/src/cpu/inner_product_cpu.cpp

CMakeFiles/inner_product_gpu.dir/src/cpu/inner_product_cpu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/inner_product_gpu.dir/src/cpu/inner_product_cpu.cpp.i"
	/home/monsterharry/Android/Sdk/ndk/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android28 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/monsterharry/Documents/sparse_gemv/src/cpu/inner_product_cpu.cpp > CMakeFiles/inner_product_gpu.dir/src/cpu/inner_product_cpu.cpp.i

CMakeFiles/inner_product_gpu.dir/src/cpu/inner_product_cpu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/inner_product_gpu.dir/src/cpu/inner_product_cpu.cpp.s"
	/home/monsterharry/Android/Sdk/ndk/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android28 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/monsterharry/Documents/sparse_gemv/src/cpu/inner_product_cpu.cpp -o CMakeFiles/inner_product_gpu.dir/src/cpu/inner_product_cpu.cpp.s

CMakeFiles/inner_product_gpu.dir/src/gpu/inner_product_opencl.cpp.o: CMakeFiles/inner_product_gpu.dir/flags.make
CMakeFiles/inner_product_gpu.dir/src/gpu/inner_product_opencl.cpp.o: /home/monsterharry/Documents/sparse_gemv/src/gpu/inner_product_opencl.cpp
CMakeFiles/inner_product_gpu.dir/src/gpu/inner_product_opencl.cpp.o: CMakeFiles/inner_product_gpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/monsterharry/Documents/sparse_gemv/build_android/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/inner_product_gpu.dir/src/gpu/inner_product_opencl.cpp.o"
	/home/monsterharry/Android/Sdk/ndk/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android28 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/inner_product_gpu.dir/src/gpu/inner_product_opencl.cpp.o -MF CMakeFiles/inner_product_gpu.dir/src/gpu/inner_product_opencl.cpp.o.d -o CMakeFiles/inner_product_gpu.dir/src/gpu/inner_product_opencl.cpp.o -c /home/monsterharry/Documents/sparse_gemv/src/gpu/inner_product_opencl.cpp

CMakeFiles/inner_product_gpu.dir/src/gpu/inner_product_opencl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/inner_product_gpu.dir/src/gpu/inner_product_opencl.cpp.i"
	/home/monsterharry/Android/Sdk/ndk/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android28 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/monsterharry/Documents/sparse_gemv/src/gpu/inner_product_opencl.cpp > CMakeFiles/inner_product_gpu.dir/src/gpu/inner_product_opencl.cpp.i

CMakeFiles/inner_product_gpu.dir/src/gpu/inner_product_opencl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/inner_product_gpu.dir/src/gpu/inner_product_opencl.cpp.s"
	/home/monsterharry/Android/Sdk/ndk/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android28 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/monsterharry/Documents/sparse_gemv/src/gpu/inner_product_opencl.cpp -o CMakeFiles/inner_product_gpu.dir/src/gpu/inner_product_opencl.cpp.s

CMakeFiles/inner_product_gpu.dir/src/cpu/dense_gemv_cpu.cpp.o: CMakeFiles/inner_product_gpu.dir/flags.make
CMakeFiles/inner_product_gpu.dir/src/cpu/dense_gemv_cpu.cpp.o: /home/monsterharry/Documents/sparse_gemv/src/cpu/dense_gemv_cpu.cpp
CMakeFiles/inner_product_gpu.dir/src/cpu/dense_gemv_cpu.cpp.o: CMakeFiles/inner_product_gpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/monsterharry/Documents/sparse_gemv/build_android/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/inner_product_gpu.dir/src/cpu/dense_gemv_cpu.cpp.o"
	/home/monsterharry/Android/Sdk/ndk/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android28 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/inner_product_gpu.dir/src/cpu/dense_gemv_cpu.cpp.o -MF CMakeFiles/inner_product_gpu.dir/src/cpu/dense_gemv_cpu.cpp.o.d -o CMakeFiles/inner_product_gpu.dir/src/cpu/dense_gemv_cpu.cpp.o -c /home/monsterharry/Documents/sparse_gemv/src/cpu/dense_gemv_cpu.cpp

CMakeFiles/inner_product_gpu.dir/src/cpu/dense_gemv_cpu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/inner_product_gpu.dir/src/cpu/dense_gemv_cpu.cpp.i"
	/home/monsterharry/Android/Sdk/ndk/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android28 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/monsterharry/Documents/sparse_gemv/src/cpu/dense_gemv_cpu.cpp > CMakeFiles/inner_product_gpu.dir/src/cpu/dense_gemv_cpu.cpp.i

CMakeFiles/inner_product_gpu.dir/src/cpu/dense_gemv_cpu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/inner_product_gpu.dir/src/cpu/dense_gemv_cpu.cpp.s"
	/home/monsterharry/Android/Sdk/ndk/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android28 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/monsterharry/Documents/sparse_gemv/src/cpu/dense_gemv_cpu.cpp -o CMakeFiles/inner_product_gpu.dir/src/cpu/dense_gemv_cpu.cpp.s

CMakeFiles/inner_product_gpu.dir/src/gpu/dense_gemv_opencl.cpp.o: CMakeFiles/inner_product_gpu.dir/flags.make
CMakeFiles/inner_product_gpu.dir/src/gpu/dense_gemv_opencl.cpp.o: /home/monsterharry/Documents/sparse_gemv/src/gpu/dense_gemv_opencl.cpp
CMakeFiles/inner_product_gpu.dir/src/gpu/dense_gemv_opencl.cpp.o: CMakeFiles/inner_product_gpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/monsterharry/Documents/sparse_gemv/build_android/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/inner_product_gpu.dir/src/gpu/dense_gemv_opencl.cpp.o"
	/home/monsterharry/Android/Sdk/ndk/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android28 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/inner_product_gpu.dir/src/gpu/dense_gemv_opencl.cpp.o -MF CMakeFiles/inner_product_gpu.dir/src/gpu/dense_gemv_opencl.cpp.o.d -o CMakeFiles/inner_product_gpu.dir/src/gpu/dense_gemv_opencl.cpp.o -c /home/monsterharry/Documents/sparse_gemv/src/gpu/dense_gemv_opencl.cpp

CMakeFiles/inner_product_gpu.dir/src/gpu/dense_gemv_opencl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/inner_product_gpu.dir/src/gpu/dense_gemv_opencl.cpp.i"
	/home/monsterharry/Android/Sdk/ndk/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android28 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/monsterharry/Documents/sparse_gemv/src/gpu/dense_gemv_opencl.cpp > CMakeFiles/inner_product_gpu.dir/src/gpu/dense_gemv_opencl.cpp.i

CMakeFiles/inner_product_gpu.dir/src/gpu/dense_gemv_opencl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/inner_product_gpu.dir/src/gpu/dense_gemv_opencl.cpp.s"
	/home/monsterharry/Android/Sdk/ndk/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android28 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/monsterharry/Documents/sparse_gemv/src/gpu/dense_gemv_opencl.cpp -o CMakeFiles/inner_product_gpu.dir/src/gpu/dense_gemv_opencl.cpp.s

# Object files for target inner_product_gpu
inner_product_gpu_OBJECTS = \
"CMakeFiles/inner_product_gpu.dir/src/main.cpp.o" \
"CMakeFiles/inner_product_gpu.dir/src/cpu/inner_product_cpu.cpp.o" \
"CMakeFiles/inner_product_gpu.dir/src/gpu/inner_product_opencl.cpp.o" \
"CMakeFiles/inner_product_gpu.dir/src/cpu/dense_gemv_cpu.cpp.o" \
"CMakeFiles/inner_product_gpu.dir/src/gpu/dense_gemv_opencl.cpp.o"

# External object files for target inner_product_gpu
inner_product_gpu_EXTERNAL_OBJECTS =

inner_product_gpu: CMakeFiles/inner_product_gpu.dir/src/main.cpp.o
inner_product_gpu: CMakeFiles/inner_product_gpu.dir/src/cpu/inner_product_cpu.cpp.o
inner_product_gpu: CMakeFiles/inner_product_gpu.dir/src/gpu/inner_product_opencl.cpp.o
inner_product_gpu: CMakeFiles/inner_product_gpu.dir/src/cpu/dense_gemv_cpu.cpp.o
inner_product_gpu: CMakeFiles/inner_product_gpu.dir/src/gpu/dense_gemv_opencl.cpp.o
inner_product_gpu: CMakeFiles/inner_product_gpu.dir/build.make
inner_product_gpu: /home/monsterharry/Documents/sparse_gemv/libOpenCL.so
inner_product_gpu: /home/monsterharry/Documents/OpenBLAS/install-android-arm64v8a/lib/libopenblas.so
inner_product_gpu: CMakeFiles/inner_product_gpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/monsterharry/Documents/sparse_gemv/build_android/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable inner_product_gpu"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/inner_product_gpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/inner_product_gpu.dir/build: inner_product_gpu
.PHONY : CMakeFiles/inner_product_gpu.dir/build

CMakeFiles/inner_product_gpu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/inner_product_gpu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/inner_product_gpu.dir/clean

CMakeFiles/inner_product_gpu.dir/depend:
	cd /home/monsterharry/Documents/sparse_gemv/build_android && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/monsterharry/Documents/sparse_gemv /home/monsterharry/Documents/sparse_gemv /home/monsterharry/Documents/sparse_gemv/build_android /home/monsterharry/Documents/sparse_gemv/build_android /home/monsterharry/Documents/sparse_gemv/build_android/CMakeFiles/inner_product_gpu.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/inner_product_gpu.dir/depend


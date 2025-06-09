# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name. The project name can be accessed via ${ PROJECT_NAME},
# Since this is the top level CMakeLists.txt, the project name is also accessible
# with ${CMAKE_PROJECT_NAME} (both CMake variables are in-sync within the top level
# build script scope).
project("demo")

file(GLOB_RECURSE SRC_LIST include/opencv2/*.cpp include/opencv2/*.c include/opencv2/*.cc include/opencv2/*.hpp)
file(GLOB_RECURSE SRC_LIST include/rga/*.cpp include/rga/*.c include/rga/*.cc include/rga/*.hpp)
file(GLOB_RECURSE SRC_LIST include/*.cpp include/*.c include/*.cc include/*.hpp)
include_directories(src/main/cpp/include)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
#
# In this top level CMakeLists.txt, ${CMAKE_PROJECT_NAME} is used to define
# the target library name; in the sub-module's CMakeLists.txt, ${PROJECT_NAME}
# is preferred for the same purpose.
#
# In order to load a library into your app from Java/Kotlin, you must call
# System.loadLibrary() and pass the name of the library defined here;
# for GameActivity/NativeActivity derived applications, the same library name must be
# used in the AndroidManifest.xml file.
add_library(${CMAKE_PROJECT_NAME} SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
		native-lib.cc
        include/lpc.cc
		include/lpd.cc
		include/lpr.cc
		include/postprocess.cc
        ${SRC_LIST}
)

target_include_directories(demo
		PUBLIC
		include
		include/opencv2
		include/rga
		include/stb)
# Specifies libraries CMake should link to your target library. You
# can link libraries from various origins, such as libraries defined in this
# build script, prebuilt third-party libraries, or Android system libraries.
target_link_libraries(${CMAKE_PROJECT_NAME}
        # List libraries link to the target library
        android
		${CMAKE_SOURCE_DIR}/${ANDROID_ABI}/libc++_shared.so
		${CMAKE_SOURCE_DIR}/${ANDROID_ABI}/libopencv_java4.so
		${CMAKE_SOURCE_DIR}/${ANDROID_ABI}/librknnrt.so
		${CMAKE_SOURCE_DIR}/${ANDROID_ABI}/librga.so
        log)
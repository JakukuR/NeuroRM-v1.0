CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(yoloXArmor)
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_BUILD_TYPE RELEASE)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPATH=\"\\\"${PROJECT_SOURCE_DIR}\\\"\"")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D${CMAKE_SYSTEM_NAME}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
SET(CMAKE_PREFIX_PATH /opt/intel/openvino_2022/runtime/cmake) # 设置 OpenVino路径(以实际安装路径为准)
SET(BIN_NAME "run")


add_definitions(-D _GLIBCXX_USE_CXX11_ABI=1)
add_compile_definitions(PROJECT_DIR="${PROJECT_SOURCE_DIR}")

FIND_PACKAGE(OpenVINO REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(OpenCV 4 REQUIRED)
FIND_PACKAGE(fmt REQUIRED)
FIND_PACKAGE(Threads)

# MVS MVC
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/Hikvision/include)
aux_source_directory(${PROJECT_SOURCE_DIR}/Hikvision/include HIK)

# MVS LIB
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/Hikvision/MVS/lib)
LINK_LIBRARIES(X11 MvCameraControl)

INCLUDE_DIRECTORIES(${InferenceEngine_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${FFTW3_INCLUDE_DIR})

LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/detector)
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/general)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/detector/include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/general/include)

FILE(GLOB_RECURSE sourcefiles "detector/src/*.cpp" "general/src/*.cpp" "Hikvision/*.cpp" )
ADD_EXECUTABLE(${BIN_NAME} main.cpp ${sourcefiles} )

SET_TARGET_PROPERTIES(${BIN_NAME} PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${PROJECT_SOURCE_DIR}/Hikvision/MVS/lib"
)

TARGET_LINK_LIBRARIES(${BIN_NAME} ${CMAKE_THREAD_LIBS_INIT}
                                  ${HIK}
                                  ${OpenCV_LIBS}
                                  fmt::fmt
                                  jsoncpp
                                  ${Eigen3_LIBRARIES}
                                  openvino::runtime
                                  MvCameraControl       # 海康相机控制库
                                  GCBase_gcc421_v3_0    # 显式添加 GenICam 基础库
                                  X11                   # 图形界面依赖
                                  )


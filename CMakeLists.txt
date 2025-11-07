cmake_minimum_required(VERSION 3.18)

set(app vk_black_hole)
project(${app})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SHADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(EXTERN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern")
set(RESOURCES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resources")

set(src
	src/main.cpp
	src/vma.cpp
    src/stb_image.c
    src/stb_image_write.c
)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/local_dirs.h.in" "${CMAKE_CURRENT_SOURCE_DIR}/src/local_dirs.h" @ONLY)

add_executable(${app} ${src})

set_property(TARGET ${app} PROPERTY CXX_STANDARD 20)

add_subdirectory(${EXTERN_DIR}/VulkanMemoryAllocator)
target_link_libraries(${app} VulkanMemoryAllocator)

add_subdirectory(${EXTERN_DIR}/vk-bootstrap)
target_link_libraries(${app} vk-bootstrap::vk-bootstrap)

find_package(Vulkan)
target_link_libraries(${app} Vulkan::Vulkan)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/black_hole.spv
    COMMAND slangc ${SHADERS_DIR}/black_hole.slang -target spirv -o ${CMAKE_CURRENT_BINARY_DIR}/black_hole.spv
    DEPENDS ${SHADERS_DIR}/black_hole.slang
    COMMENT "Building black_hole.spv"
)

set(SHADER_BINS
    ${CMAKE_CURRENT_BINARY_DIR}/black_hole.spv
)

add_custom_target(
    Shaders ALL
    DEPENDS ${SHADER_BINS}
)

add_dependencies(${app} Shaders)

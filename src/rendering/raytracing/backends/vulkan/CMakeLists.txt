set(RL_TOOLS_VULKAN_DEVICE_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/device.comp)
set(RL_TOOLS_VULKAN_SPIRV_DIR ${CMAKE_CURRENT_BINARY_DIR}/spirv)

set(RL_TOOLS_VULKAN_SPIRV_HEADERS "")
foreach(RL_TOOLS_VULKAN_VARIANT rgb depth collision segmentation normals flow resolve)
    string(TOUPPER ${RL_TOOLS_VULKAN_VARIANT} RL_TOOLS_VULKAN_VARIANT_UPPER)
    set(RL_TOOLS_VULKAN_SPIRV_HEADER ${RL_TOOLS_VULKAN_SPIRV_DIR}/device_spirv_${RL_TOOLS_VULKAN_VARIANT}.h)
    add_custom_command(
        OUTPUT ${RL_TOOLS_VULKAN_SPIRV_HEADER}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${RL_TOOLS_VULKAN_SPIRV_DIR}
        COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.2
            -DENTRY_${RL_TOOLS_VULKAN_VARIANT_UPPER}=1
            --vn rl_tools_rendering_raytracing_vulkan_device_spirv_${RL_TOOLS_VULKAN_VARIANT}_words
            -o ${RL_TOOLS_VULKAN_SPIRV_HEADER}
            ${RL_TOOLS_VULKAN_DEVICE_SOURCE}
        DEPENDS ${RL_TOOLS_VULKAN_DEVICE_SOURCE}
        COMMENT "Compiling Vulkan device program (${RL_TOOLS_VULKAN_VARIANT})"
    )
    list(APPEND RL_TOOLS_VULKAN_SPIRV_HEADERS ${RL_TOOLS_VULKAN_SPIRV_HEADER})
endforeach()

add_library(rendering_raytracing_vulkan STATIC device_spirv.cpp ${RL_TOOLS_VULKAN_SPIRV_HEADERS})
target_include_directories(rendering_raytracing_vulkan PRIVATE ${RL_TOOLS_VULKAN_SPIRV_DIR})
target_link_libraries(rendering_raytracing_vulkan PUBLIC Vulkan::Vulkan)
set_target_properties(rendering_raytracing_vulkan PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

add_library(rendering_raytracing_backend INTERFACE)
target_link_libraries(rendering_raytracing_backend INTERFACE rendering_raytracing_vulkan assimp::assimp)

add_library(rendering_raytracing_backend_depth INTERFACE)
target_link_libraries(rendering_raytracing_backend_depth INTERFACE rendering_raytracing_vulkan assimp::assimp)

add_library(rendering_raytracing_backend_segmentation INTERFACE)
target_link_libraries(rendering_raytracing_backend_segmentation INTERFACE rendering_raytracing_vulkan assimp::assimp)

add_library(rendering_raytracing_backend_normals INTERFACE)
target_link_libraries(rendering_raytracing_backend_normals INTERFACE rendering_raytracing_vulkan assimp::assimp)

add_plugin(NeuralNoiseReduction)

target_sources(NeuralNoiseReduction PRIVATE
    Common.h
    Common.cpp
    Common.slang
    NeuralNoiseReduction.cpp
    NeuralNoiseReduction.h
    RenderContextUtils.h
    RenderContextUtils.cpp
    RenderingComponent.h
    RenderingComponent.cpp
    ShaderUtils.h
    ShaderUtils.cpp
    SimpleKernel.h
    SimpleKernel.cpp
    SimpleKernel.ps.slang
    SubrenderGraph.h
    SubrenderGraph.cpp
    TextureData.h
    TextureData.cpp
    TextureReflecter.h
    TextureReflecter.cpp
)

target_source_group(NeuralNoiseReduction "RenderPasses")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_link_libraries(NeuralNoiseReduction PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET NeuralNoiseReduction PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET NeuralNoiseReduction
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:NeuralNoiseReduction>)
endif (MSVC)

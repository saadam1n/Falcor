add_plugin(NeuralNoiseReduction)

target_sources(NeuralNoiseReduction PRIVATE
    Common.h
    Common.cpp
    Common.slang
    NeuralNoiseReduction.cpp
    NeuralNoiseReduction.h
    Optimizer.h
    Optimizer.cpp
    ParameterReflecter.h
    ParameterReflecter.cpp
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

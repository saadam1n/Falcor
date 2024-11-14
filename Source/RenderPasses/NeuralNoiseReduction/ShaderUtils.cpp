#include "ShaderUtils.h"

DefineList appendPassTypeToDefineList(const DefineList& dl, NeuralNetPassType nnpt)
{
    DefineList dl2 = dl;

    if (nnpt == NEURAL_NET_PASS_TYPE_FORWARD)
    {
        dl2.add("FWD_PASS");
    }
    else if (nnpt == NEURAL_NET_PASS_TYPE_BACKWARD)
    {
        dl2.add("BWD_PASS");
    }

    return dl2;
}

ref<FullScreenPass> createFullscreenPassAndDumpIR(
    ref<Device> pDevice,
    const std::string& path,
    NeuralNetPassType nnpt,
    const DefineList& dl
)
{
    ProgramDesc desc;
    desc.compilerFlags |= SlangCompilerFlags::DumpIntermediates;
    desc.addShaderLibrary(path).psEntry("main");
    return FullScreenPass::create(pDevice, desc, appendPassTypeToDefineList(dl, nnpt));
}

ref<ComputePass> createComputePassAndDumpIR(
    ref<Device> pDevice,
    const std::string& path,
    NeuralNetPassType nnpt,
    const DefineList& dl
)
{
    ProgramDesc desc;
    desc.compilerFlags |= SlangCompilerFlags::DumpIntermediates;
    desc.addShaderLibrary(path).csEntry("main");
    return ComputePass::create(pDevice, desc, appendPassTypeToDefineList(dl, nnpt));
}

#pragma once

#include "Falcor.h"
#include "RenderGraph/RenderPass.h"

#include "Common.h"

using namespace Falcor;

#include <vector>

// I call it vertex because of graph terminology
// A texture will be a endpoint/startpoint for some edge in the rendergraph
struct TextureVertex
{
    std::string name;
    int width = sParams.patchWidth;
    int height = sParams.patchHeight;
    ResourceFormat fmt = ResourceFormat::RGBA32Float;
    ResourceBindFlags flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget;
};

class TextureReflecter
{
public:
    TextureVertex& addInput(const std::string& name);
    TextureVertex& addOutput(const std::string& name);

    const std::vector<TextureVertex>& getInputs();
    const std::vector<TextureVertex>& getOutputs();

private:
    void throwIfAlreadyPresent(const std::vector<TextureVertex>& v, const std::string& s);

    std::vector<TextureVertex> mInputList;
    std::vector<TextureVertex> mOutputList;
};

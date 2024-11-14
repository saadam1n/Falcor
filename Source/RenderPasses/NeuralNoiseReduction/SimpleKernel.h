#pragma once

#include "Falcor.h"
#include "RenderGraph/RenderPass.h"

#include "RenderingComponent.h"

using namespace Falcor;

namespace
{
const char* kSimpleKernelInput = "src";
const char* kSimpleKernelOutput = "dst";
const char* kSimpleKernelShader = "RenderPasses/NeuralNoiseReduction/SimpleKernel.ps.slang";
}

class SimpleKernel : public RenderingComponent
{
public:
    SimpleKernel(ref<Device> pDevice);
    void allocateFbos();

    virtual ~SimpleKernel() override;

    virtual void reflect(TextureReflecter& reflecter) override;

    virtual void forward(RenderContext* pRenderContext, const TextureData& textureData) override;
    virtual void backward(RenderContext* pRenderContext, const TextureData& textureData) override;

private:
    ref<FullScreenPass> mpBlurFilter;
    ref<Fbo> mpBlurringFbo;
};

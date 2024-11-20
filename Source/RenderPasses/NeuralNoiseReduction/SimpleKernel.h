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

#define KERNEL_DIM 15

class SimpleKernel : public RenderingComponent
{
public:
    SimpleKernel(ref<Device> pDevice);
    void allocateFbos();

    virtual ~SimpleKernel() override;

    virtual void reflectTextures(TextureReflecter& reflecter) override;

    virtual void forward(RenderContext* pRenderContext, const TextureData& textureData) override;
    virtual void backward(RenderContext* pRenderContext, const TextureData& textureData) override;

private:
    float mKernel[KERNEL_DIM][KERNEL_DIM];


    ref<FullScreenPass> mpBlurFilter;
    ref<Fbo> mpBlurringFbo;

};

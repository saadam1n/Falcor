#include "SimpleKernel.h"
#include "RenderContextUtils.h"

SimpleKernel::SimpleKernel(ref<Device> pDevice) : RenderingComponent(pDevice)
{
    mpBlurFilter = createFullscreenPassAndDumpIR(mpDevice, kSimpleKernelShader, NEURAL_NET_PASS_TYPE_FORWARD);
    allocateFbos();
}

void SimpleKernel::allocateFbos()
{
    {
        Fbo::Desc desc;
        desc.setSampleCount(0);
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float);
        mpBlurringFbo = Fbo::create2D(mpDevice, sParams.patchWidth, sParams.patchHeight, desc);
    }
}

SimpleKernel::~SimpleKernel() {

}

void SimpleKernel::reflect(TextureReflecter& reflecter)
{
    reflecter.addInput(kSimpleKernelInput);
    reflecter.addOutput(kSimpleKernelOutput);
}

void SimpleKernel::forward(RenderContext* pRenderContext, const TextureData& textureData)
{
    auto pSrc = textureData.getTexture(kSimpleKernelInput);
    auto pDst = textureData.getTexture(kSimpleKernelOutput);

    auto perImageCB = mpBlurFilter->getRootVar()["PerImageCB"];

    perImageCB["src"] = pSrc;

    mpBlurFilter->execute(pRenderContext, mpBlurringFbo);

    blitTextures(pRenderContext, mpBlurringFbo->getColorTexture(0), pDst);
}

void SimpleKernel::backward(RenderContext* pRenderContext, const TextureData& textureData)
{

}



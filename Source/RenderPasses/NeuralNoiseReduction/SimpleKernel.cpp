#include "SimpleKernel.h"
#include "RenderContextUtils.h"

SimpleKernel::SimpleKernel(ref<Device> pDevice) : RenderingComponent(pDevice)
{
    mpBlurFilter = createFullscreenPassAndDumpIR(mpDevice, kSimpleKernelShader, NEURAL_NET_PASS_TYPE_FORWARD);

    float totalWeight = 0.0f;
    float var = 0.1f;
    for (int i = 0; i < KERNEL_DIM; i++)
    {
        for (int j = 0; j < KERNEL_DIM; j++)
        {
            int ydist = i - KERNEL_DIM / 2;
            int xdist = j - KERNEL_DIM / 2;
            mKernel[i][j] = exp(var * -(xdist * xdist + ydist * ydist));

            totalWeight += mKernel[i][j];
        }
    }

    for (int i = 0; i < KERNEL_DIM; i++)
    {
        for (int j = 0; j < KERNEL_DIM; j++)
        {
            mKernel[i][j] /= totalWeight;
        }
    }

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

SimpleKernel::~SimpleKernel() {}

void SimpleKernel::reflect(TextureReflecter& reflecter)
{
    reflecter.addInput(kSimpleKernelInput);
    reflecter.addOutput(kSimpleKernelOutput).setPredefinedLocation(mpBlurringFbo->getColorTexture(0));
}

void SimpleKernel::forward(RenderContext* pRenderContext, const TextureData& textureData)
{
    auto pSrc = textureData.getTexture(kSimpleKernelInput);

    auto perImageCB = mpBlurFilter->getRootVar()["PerImageCB"];

    perImageCB["src"] = pSrc;
    perImageCB["kernel"].setBlob(mKernel);

    // blurring fbo points directly to output
    mpBlurFilter->execute(pRenderContext, mpBlurringFbo);

    ASSERT_TEXTURE_IS_OUTPUT(textureData, mpBlurringFbo->getColorTexture(0), kSimpleKernelOutput);
}

void SimpleKernel::backward(RenderContext* pRenderContext, const TextureData& textureData)
{

}



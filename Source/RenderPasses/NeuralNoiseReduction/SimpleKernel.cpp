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
    reflecter.addOutput(kSimpleKernelOutput).setPredefinedLocation(mpBlurringFbo->getColorTexture(0));

    std::cout << "Predefined location is " << mpBlurringFbo->getColorTexture(0) << std::endl;
}

void SimpleKernel::forward(RenderContext* pRenderContext, const TextureData& textureData)
{
    auto pSrc = textureData.getTexture(kSimpleKernelInput);

    auto perImageCB = mpBlurFilter->getRootVar()["PerImageCB"];

    perImageCB["src"] = pSrc;

    // blurring fbo points directly to output
    mpBlurFilter->execute(pRenderContext, mpBlurringFbo);

    ref<Texture> np = nullptr;
    std::cout << mpBlurringFbo->getColorTexture(0) << std::endl;
    std::cout << textureData.getTexture(kSimpleKernelOutput) << std::endl;
    std::cout << np << std::endl;
    std::cout << "========\n";
    std::cout << mpBlurringFbo->getColorTexture(0).get() << std::endl;
    std::cout << textureData.getTexture(kSimpleKernelOutput).get() << std::endl;
    std::cout << np.get() << std::endl;

    ASSERT_TEXTURE_IS_OUTPUT(textureData, mpBlurringFbo->getColorTexture(0), kSimpleKernelOutput);
}

void SimpleKernel::backward(RenderContext* pRenderContext, const TextureData& textureData)
{

}



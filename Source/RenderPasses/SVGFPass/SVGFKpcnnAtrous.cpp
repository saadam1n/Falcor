#include "SVGFKpcnnAtrous.h"

SVGFKpcnnAtrousSubpass::SVGFKpcnnAtrousSubpass(ref<Device> pDevice, ref<SVGFUtilitySet> pUtilities, ref<FilterParameterReflector> pParameterReflector)
    : mpDevice(pDevice), mpUtilities(pUtilities), mpParameterReflector(pParameterReflector)
{
    mpEvaluatePass = ComputePass::create(mpDevice, kKpcnnAtrousShaderS);
    mpBackPropagatePass = ComputePass::create(mpDevice, kKpcnnAtrousShaderD);

    // create some test stuff
    mpTestIllum = mpDevice->createTexture2D(5, 5, ResourceFormat::RGBA32Float);
    mpTestNormalDepth = mpDevice->createTexture2D(5, 5, ResourceFormat::RGBA32Float);
    mpTestOutput = mpDevice->createTexture2D(5, 5, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
}

void SVGFKpcnnAtrousSubpass::allocateFbos(uint2 dim, RenderContext* pRenderContext)
{}

void SVGFKpcnnAtrousSubpass::runTest(RenderContext* pRenderContext)
{
    // run a simple 5x5 patch and verify that our output is in line with what we would expect

    // set the test data
    float4 testIllumData[5][5];
    float4 testNormalData[5][5];
    for (int y = 0; y < 5; y++)
    {
        for (int x = 0; x < 5; x++)
        {
            testIllumData[y][x] = float4(1.0f, 0.0f, 0.0f, 0.0f);
            testNormalData[y][x] = float4(0.0f);
        }
    }

    pRenderContext->updateTextureData(mpTestIllum.get(), (const void*)testIllumData);
    pRenderContext->updateTextureData(mpTestNormalDepth.get(), (const void*)testNormalData);

    auto perImageCB = mpEvaluatePass->getRootVar()["PerImageCB"];
    perImageCB["gIllumination"] = mpTestIllum;
    perImageCB["gLinearZAndNormal"] = mpTestNormalDepth;
    perImageCB["gFiltered"] = mpTestOutput;
    perImageCB["gStepSize"] = uint2(1, 1);

    mpEvaluatePass->execute(pRenderContext, uint3(1, 1, 25));

    // now download test data
    auto outputBitmap = pRenderContext->readTextureSubresource(mpTestOutput.get(), 0);
    float4(*filteredImage)[5] = (float4(*)[5])outputBitmap.data(); // uh super weird syntax I do not understand

    for (int y = 0; y < 5; y++)
    {
        for (int x = 0; x < 5; x++)
        {
            std::cout << filteredImage[y][x].r << "\t";
        }
        std::cout << "\n";
    }
    std::cout.flush();
}

void SVGFKpcnnAtrousSubpass::computeEvaluation(RenderContext* pRenderContext, SVGFRenderData& svgfrd, bool updateInternalBuffers)
{
}

void SVGFKpcnnAtrousSubpass::computeBackPropagation(RenderContext* pRenderContext, SVGFRenderData& svgfrd)
{
}

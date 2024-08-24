#include "SVGFKpcnnAtrous.h"

SVGFKpcnnAtrousSubpass::SVGFKpcnnAtrousSubpass(ref<Device> pDevice, ref<SVGFUtilitySet> pUtilities, ref<FilterParameterReflector> pParameterReflector)
    : mpDevice(pDevice), mpUtilities(pUtilities), mpParameterReflector(pParameterReflector)
{
    mpEvaluatePass = mpUtilities->createComputePassAndDumpIR(kKpcnnAtrousShaderS);
    mpBackPropagatePass = mpUtilities->createComputePassAndDumpIR(kKpcnnAtrousShaderD);

    // create some test stuff
    mpTestIllum = mpDevice->createTexture2D(5, 5, ResourceFormat::RGBA32Float);
    mpTestNormalDepth = mpDevice->createTexture2D(5, 5, ResourceFormat::RGBA32Float);
    mpTestOutput = mpDevice->createTexture2D(5, 5, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
}

void SVGFKpcnnAtrousSubpass::allocateFbos(uint2 dim, RenderContext* pRenderContext)
{}

void SVGFKpcnnAtrousSubpass::runTest(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "KPCNN Test");
    // run a simple 5x5 patch and verify that our output is in line with what we would expect
    // 
    // set the test data
    for (int y = 0; y < 5; y++)
    {
        for (int x = 0; x < 5; x++)
        {
            mpTestIllumData[y][x] = float4(1.0f, 0.0f, 0.0f, 0.0f);
            mpTestNormalData[y][x] = float4(0.0f);
        }
    }
    pRenderContext->updateTextureData(mpTestIllum.get(), (const void*)mpTestIllumData);
    pRenderContext->updateTextureData(mpTestNormalDepth.get(), (const void*)mpTestNormalData);

    // set up our variables
    for (int k = 0; k < 8; k++)
    {
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                mpPostconvKernels[k].weights[i][j] = 1.0f / 25.0f;
            }
        }
    }

    auto perImageCB = mpEvaluatePass->getRootVar()["PerImageCB"];
    perImageCB["gIllumination"] = mpTestIllum;
    perImageCB["gLinearZAndNormal"] = mpTestNormalDepth;
    perImageCB["gFiltered"] = mpTestOutput;
    perImageCB["gStepSize"] = uint2(1, 1);
    perImageCB["postconv"].setBlob(mpPostconvKernels);


    mpEvaluatePass->execute(pRenderContext, uint3(1, 1, 25));

    // now download test data
    auto outputBitmap = pRenderContext->readTextureSubresource(mpTestOutput.get(), 0);
    float4(*filteredImage)[5] = (float4(*)[5])outputBitmap.data(); // uh super weird syntax I do not understand

    for (int y = -1; y < 5; y++)
    {
        for (int x = -1; x < 5; x++)
        {
            if (x == -1 && y == -1)
            {
                std::cout << "y/x\t";
            }
            else if (y == -1)
            {
                std::cout << x << "\t";
            }
            else if (x == -1)
            {
                std::cout << y << "\t";
            }
            else
            {
                std::cout << filteredImage[y][x].r << "," << filteredImage[y][x].g << "\t";
            }

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

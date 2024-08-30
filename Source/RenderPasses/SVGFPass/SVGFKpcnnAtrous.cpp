#include "SVGFKpcnnAtrous.h"

SVGFKpcnnAtrousSubpass::SVGFKpcnnAtrousSubpass(ref<Device> pDevice, ref<SVGFUtilitySet> pUtilities, ref<FilterParameterReflector> pParameterReflector)
    : mpDevice(pDevice), mpUtilities(pUtilities), mpParameterReflector(pParameterReflector)
{
    mpEvaluatePass = mpUtilities->createComputePassAndDumpIR(kKpcnnAtrousShaderS);
    mpBackPropagatePass = mpUtilities->createComputePassAndDumpIR(kKpcnnAtrousShaderD);

    mpPixelDebug = std::make_unique<PixelDebug>(mpDevice);
    mpPixelDebug->enable();

    // create some test stuff
    mpTestIllum = mpDevice->createTexture2D(kMapDim, kMapDim, ResourceFormat::RGBA32Float);
    mpTestNormalDepth = mpDevice->createTexture2D(kMapDim, kMapDim, ResourceFormat::RGBA32Float);
    mpTestOutput = mpDevice->createTexture2D(
        kMapDim, kMapDim, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
}

void SVGFKpcnnAtrousSubpass::allocateFbos(uint2 dim, RenderContext* pRenderContext)
{}

void SVGFKpcnnAtrousSubpass::runTest(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "KPCNN Test");
    // run a simple 5x5 patch and verify that our output is in line with what we would expect
    // 
    // set the test data

    float4 tempTestIllumData[5][5] = {
        {float4(0.0f, 0.0f, 0.0f, 0.0f),
         float4(0.0f, 0.0f, 0.0f, 0.0f),
         float4(0.0f, 0.0f, 0.0f, 0.0f),
         float4(0.0f, 0.0f, 0.0f, 0.0f),
         float4(0.0f, 0.0f, 0.0f, 0.0f)},
        {float4(0.0f, 0.0f, 0.0f, 0.0f),
         float4(1.0f, 0.0f, 0.0f, 0.0f),
         float4(1.0f, 0.0f, 0.0f, 0.0f),
         float4(0.0f, 0.0f, 0.0f, 0.0f),
         float4(0.0f, 0.0f, 0.0f, 0.0f)},
        {float4(0.0f, 0.0f, 0.0f, 0.0f),
         float4(1.0f, 0.0f, 0.0f, 0.0f),
         float4(1.0f, -1.0f, 0.0f, 0.0f),
         float4(0.0f, -1.0f, 0.0f, 0.0f),
         float4(0.0f, 0.0f, 0.0f, 0.0f)},
        {float4(0.0f, 0.0f, 0.0f, 0.0f),
         float4(0.0f, 0.0f, 0.0f, 0.0f),
         float4(0.0f, -1.0f, 0.0f, 0.0f),
         float4(0.0f, -1.0f, 0.0f, 0.0f),
         float4(0.0f, 0.0f, 0.0f, 0.0f)},
        {float4(0.0f, 0.0f, 0.0f, 0.0f),
         float4(0.0f, 0.0f, 0.0f, 0.0f),
         float4(0.0f, 0.0f, 0.0f, 0.0f),
         float4(0.0f, 0.0f, 0.0f, 0.0f),
         float4(0.0f, 0.0f, 0.0f, 0.0f)},
    };

    for (int y = 0; y < kMapDim; y++)
    {
        for (int x = 0; x < kMapDim; x++)
        {
            mTestIllumData[y][x] = tempTestIllumData[y][x];
            mTestNormalData[y][x] = float4(0.0f);
        }
    }
    pRenderContext->updateTextureData(mpTestIllum.get(), (const void*)mTestIllumData);
    pRenderContext->updateTextureData(mpTestNormalDepth.get(), (const void*)mTestNormalData);

    // set up our variables
    for (int k = 0; k < kOutputMapsPerLayer * kNumLayers; k++)
    {
        for (int s = 0; s < kOutputMapsPerLayer; s++)
        {
            for (int i = 0; i < kKernelDim; i++)
            {
                for (int j = 0; j < kKernelDim; j++)
                {
                    mKernels[k].weights[s][i][j] = 1.0f / (kKernelDim * kKernelDim);
                }
                mKernels[k].bias = 0.0f;
            }
        }
    }

    for (int k = 0; k < kOutputMapsPerLayer; k++)
    {
        for (int i = 0; i < kMapDim; i++)
        {
            for (int j = 0; j < kMapDim; j++)
            {
                mPostconvKernels[k].weights[i][j] = 1.0f / (kMapDim * kMapDim);
            }
        }
    }

    auto perImageCB = mpEvaluatePass->getRootVar()["PerImageCB"];
    perImageCB["gIllumination"] = mpTestIllum;
    perImageCB["gLinearZAndNormal"] = mpTestNormalDepth;
    perImageCB["gFiltered"] = mpTestOutput;
    perImageCB["gStepSize"] = uint2(1, 1);
    perImageCB["postconv"].setBlob(mPostconvKernels);
    perImageCB["kernels"].setBlob(mKernels);

    mpPixelDebug->beginFrame(pRenderContext, uint2(kMapDim, kMapDim));
    mpPixelDebug->prepareProgram(mpEvaluatePass->getProgram(), mpEvaluatePass->getRootVar());
    mpEvaluatePass->execute(pRenderContext, uint3(1, 1, 25));
    mpPixelDebug->endFrame(pRenderContext);

    // now download test data
    auto outputBitmap = pRenderContext->readTextureSubresource(mpTestOutput.get(), 0);
    float4(*filteredImage)[kMapDim] = (float4(*)[kMapDim])outputBitmap.data(); // uh super weird syntax I do not understand

    std::cout << "GPU Test Result:\n";
    print_test_result(filteredImage);

    std::cout << "\n\n\n";

    simulate_kpcnn();

    std::cout.flush();
}

void SVGFKpcnnAtrousSubpass::computeEvaluation(RenderContext* pRenderContext, SVGFRenderData& svgfrd, bool updateInternalBuffers)
{
}

void SVGFKpcnnAtrousSubpass::computeBackPropagation(RenderContext* pRenderContext, SVGFRenderData& svgfrd)
{}


float SVGFKpcnnAtrousSubpass::ConvolutionKernel::fetch_weight(const int map, const int x, const int y)
{
    const int linearIdx = kKernelDim * kKernelDim * map + kKernelDim * y + x;
    const int elemIdx = linearIdx / 4;
    const int chnlIdx = linearIdx % 4;
    return packed_weights[elemIdx][chnlIdx];
}

float SVGFKpcnnAtrousSubpass::PostconvolutionKernel::fetch_weight(const int x, const int y)
{
    const int linearIdx = y * kMapDim + x;
    const int elemIdx = linearIdx / 4;
    const int chnlIdx = linearIdx % 4;
    return packed_weights[elemIdx][chnlIdx];
}

float& SVGFKpcnnAtrousSubpass::ConvolutionMap::get(const int x, const int y)
{
    return m[y][x];
}

void SVGFKpcnnAtrousSubpass::printPixelDebug(const std::string& s, float x)
{
    if (mCpuPrintingEnabled)
    {
        std::cout << s << x << std::endl;
    }
}

void SVGFKpcnnAtrousSubpass::print_test_result(float4 grid[][kMapDim])
{
    for (int y = -1; y < kMapDim; y++)
    {
        for (int x = -1; x < kMapDim; x++)
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
                std::cout << grid[y][x].r << "," << grid[y][x].g << "\t";
            }
        }
        std::cout << "\n";
    }
    std::cout.flush();
}

void SVGFKpcnnAtrousSubpass::simulate_thread_group_sequentially(std::function<void(uint2)> func)
{
    for (int y = 0; y < kMapDim; y++)
    {
        for (int x = 0; x < kMapDim; x++)
        {
            uint2 offset = uint2(x, y);

            mCpuPrintingEnabled = false;
            if (offset.x == mCurrentCpuDebugPixel.x && offset.y == mCurrentCpuDebugPixel.y)
            {
                mCpuPrintingEnabled = true;
            }

            func(offset);
        }
    }
}

void SVGFKpcnnAtrousSubpass::simulate_kpcnn()
{

    // simulate each "wave" of pixels at a time

    simulate_thread_group_sequentially([&](uint2 offset) {
        for (int i = 0; i < kOutputMapsPerLayer; i++)
        {
            float writeVal;
            #if 0
            if (i < 4)
            {
                writeVal = mTestIllumData[offset.y][offset.x][i];
            }
            else if (i < 8)
            {
                writeVal = mTestNormalData[offset.y][offset.x][i - 4];
            }
            else if (i < 12)
            {
                writeVal = 0.0f; // worldPosAtCurPixel[i - 8];
            }
            else
            {
                writeVal = 0.0f;
            }
            #else
            if (i < 4)
            {
                writeVal = mTestIllumData[offset.y][offset.x][i];
            }
            else
            {
                writeVal = 0.0f;
            }
            #endif



            mRbuf[i].m[offset.y][offset.x] = writeVal;
        }
    });

    int currentReadIndex = 0;
    int currentWriteIndex = kOutputMapsPerLayer;
    int currentKernelIdx = 0;
    for (int layerIndex = 0; layerIndex < kNumLayers; layerIndex++)
    {
        ConvolutionMap tRbuf[kOutputMapsPerLayer];
        reference_convolution(currentReadIndex, currentKernelIdx, tRbuf);


        for (int outputMapIndex = 0; outputMapIndex < kOutputMapsPerLayer; outputMapIndex++)
        {
            simulate_thread_group_sequentially([&](uint2 offset) {
                clear_accumulation_area(offset, currentWriteIndex);
            });

            simulate_thread_group_sequentially([&](uint2 offset) {
                convolve_kernel(offset, currentReadIndex, currentWriteIndex, currentKernelIdx);
            });

            simulate_thread_group_sequentially([&](uint2 offset) {
                reduce_and_activate(offset, currentWriteIndex, currentKernelIdx);
            });

            currentWriteIndex++;
            currentKernelIdx++;
        }

        currentReadIndex += kOutputMapsPerLayer;

        for (int i = 0; i < kOutputMapsPerLayer; i++)
        {
            for (int y = 0; y < kMapDim; y++)
            {
                for (int x = 0; x < kMapDim; x++)
                {
                    float calc = mRbuf[(currentReadIndex + i) % kRingBufferSize].m[y][x];
                    float ref = tRbuf[i].m[y][x];

                    float err = abs(calc - ref);
                    std::cout << err << "\t";

                }
            }
        }
        std::cout << "\n";
    }

    std::cout << "Outputs are stored at index starting " << currentReadIndex << std::endl;

    float4 filteredImage[kMapDim][kMapDim];
    simulate_thread_group_sequentially([&](uint2 offset) {
        float maxRawOut = 0.0f;
        for (int i = 0; i < kNumOutputWeights; i++)
        {
            maxRawOut = std::max(maxRawOut, mRbuf[(currentReadIndex + i) % kRingBufferSize].m[offset.y][offset.x]);
        }

        float totalWeight = 0.0f;
        //std::cout << offset.x << "\t" << offset.y << "\t";
        for (int i = 0; i < kNumOutputWeights; i++)
        {
            //std::cout << mRbuf[(currentReadIndex + i) % kRingBufferSize].m[offset.y][offset.x] - maxRawOut << "\t";

            mRbuf[(currentReadIndex + i) % kRingBufferSize].m[offset.y][offset.x] =
                exp(mRbuf[(currentReadIndex + i) % kRingBufferSize].m[offset.y][offset.x] - maxRawOut);
            totalWeight += mRbuf[(currentReadIndex + i) % kRingBufferSize].m[offset.y][offset.x];
        }
        //std::cout << "\n";



        float4 convIllum = float4(0.0f);
        for (int i = 0; i < kNumOutputWeights; i++)
        {
            float4 tempAccumIllum = float4(0.0f);
            for (int y = 0; y < kMapDim; y++)
            {
                for (int x = 0; x < kMapDim; x++)
                {
                    tempAccumIllum += mPostconvKernels[i].fetch_weight(x, y) * mTestIllumData[y][x];
                }
            }

            float weight = mRbuf[(currentReadIndex + i) % kRingBufferSize].m[offset.y][offset.x] / totalWeight;

            if (i < 4)
            {
                convIllum[i] = weight;
            }
            else
            {
                break;
            }
            continue;

            convIllum += weight * tempAccumIllum;
        }

        filteredImage[offset.y][offset.x] = convIllum;
     });

    std::cout << "CPU KPCNN Result:\n";
    print_test_result(filteredImage);
}

void SVGFKpcnnAtrousSubpass::reference_convolution(int readIdx, int kernelIdx, ConvolutionMap tRbuf[])
{
    for (int dstLayer = 0; dstLayer < kOutputMapsPerLayer; dstLayer++)
    {
        for (int ydst = 0; ydst < kMapDim; ydst++)
        {
            for (int xdst = 0; xdst < kMapDim; xdst++)
            {
                float sum = mKernels[kernelIdx + dstLayer].bias;

                for (int srcLayer = 0; srcLayer < kOutputMapsPerLayer; srcLayer++)
                {
                    for (int yoff = -kKernelDistance; yoff <= kKernelDistance; yoff++)
                    {
                        for (int xoff = -kKernelDistance; xoff <= kKernelDistance; xoff++)
                        {
                            int xsrc = xdst + xoff;
                            int ysrc = ydst + yoff;

                            if (xsrc < 0 || ysrc < 0 || xsrc >= kMapDim || ysrc >= kMapDim)
                            {
                                continue;
                            }

                            float sourcev = mRbuf[(readIdx + srcLayer) % kRingBufferSize].m[ysrc][xsrc];
                            sum += mKernels[kernelIdx + dstLayer].fetch_weight(dstLayer, xoff + kKernelDistance, yoff + kKernelDistance) *
                                   sourcev;
                        }
                    }
                }


                tRbuf[dstLayer].m[ydst][xdst] = std::max(sum, 0.0f);
            }
        }
    }
}

void SVGFKpcnnAtrousSubpass::clear_accumulation_area(uint2 srcPix, int writeIdx)
{
    // first things first, we need to zero out everything in accumulation block
    for (int i = 0; i < kKernelSummationTerms; i++)
    {
        mRbuf[(writeIdx + i) % kRingBufferSize].m[srcPix.y][srcPix.x] = 0.0f;
    }
}

void SVGFKpcnnAtrousSubpass::convolve_kernel(uint2 srcPix, int readIdx, int writeIdx, int kernelIdx)
{
    for (int y = -kKernelDistance; y <= kKernelDistance; y++)
    {
        for (int x = -kKernelDistance; x <= kKernelDistance; x++)
        {
            const int2 dstPixel = int2(srcPix) + int2(x, y);
            const bool inside = (dstPixel.x >= 0 && dstPixel.y >= 0 && dstPixel.x < kMapDim && dstPixel.y < kMapDim);

            if (inside)
            {
                float sum = 0.0f;
                // now, accumulate to our target pixel
                for (int srcLayer = 0; srcLayer < kOutputMapsPerLayer; srcLayer++)
                {
                    float mapVal = mRbuf[(readIdx + srcLayer) % kRingBufferSize].m[srcPix.y][srcPix.x];
                    sum += mapVal * mKernels[kernelIdx].fetch_weight(srcLayer, x + kKernelDistance, y + kKernelDistance);
                }

                int offsetIdx = kKernelDim * (y + kKernelDistance) + (x + kKernelDistance);

                mRbuf[(writeIdx + offsetIdx) % kRingBufferSize].m[dstPixel.y][dstPixel.x] = sum;
            }
        }
    }

    // now sync for future passes
    //GroupMemoryBarrierWithGroupSync();
}

void SVGFKpcnnAtrousSubpass::reduce_and_activate(uint2 offset, int writeIdx, int kernelIdx)
{
    // no fancy parallel reduction for now, just plain "linear" accumulation
    int dstIdx = writeIdx % kRingBufferSize;
    //printPixelDebug("\tTerm  ", mRbuf[dstIdx].m[offset.y][offset.x]);

    for (int i = 1; i < kKernelSummationTerms; i++)
    {
        //printPixelDebug("\tTerm  ", mRbuf[(writeIdx + i) % kRingBufferSize].m[offset.y][offset.x]);

        mRbuf[dstIdx].m[offset.y][offset.x] += mRbuf[(writeIdx + i) % kRingBufferSize].m[offset.y][offset.x];
    }

    //printPixelDebug("Post Sum ", mRbuf[dstIdx].m[offset.y][offset.x]);

    // now apply bias
    mRbuf[dstIdx].m[offset.y][offset.x] += mKernels[kernelIdx].bias;

    //printPixelDebug("Post Bias ", mRbuf[dstIdx].m[offset.y][offset.x]);
    //printPixelDebug("\tBias", mKernels[kernelIdx].bias);

    // apply ReLU
    mRbuf[dstIdx].m[offset.y][offset.x] = std::max(mRbuf[dstIdx].m[offset.y][offset.x], 0.0f);

    //printPixelDebug("Post ReLU ", mRbuf[dstIdx].m[offset.y][offset.x]);
}

void SVGFKpcnnAtrousSubpass::renderUI(Gui::Widgets& widget)
{
    mpPixelDebug->renderUI(widget);
}



#pragma once

#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Core/Pass/FullScreenPass.h"
#include "Utils/Algorithm/ParallelReduction.h"

#include "SVGFCommon.h"

using namespace Falcor;

#define kMapDim 5
#define kNumPixels (kMapDim * kMapDim)
#define kKernelDistance 1
#define kKernelDim 3
#define kKernelSummationTerms (kKernelDim * kKernelDim)
#define kOutputMapsPerLayer 8
#define kRingBufferSize (2 * kOutputMapsPerLayer + kKernelSummationTerms)
#define kNumLayers 4
#define kNumOutputWeights 8

class SVGFKpcnnAtrousSubpass : public Object
{
public:
    SVGFKpcnnAtrousSubpass(ref<Device> pDevice, ref<SVGFUtilitySet> pUtilities, ref<FilterParameterReflector> pParameterReflector);
    void allocateFbos(uint2 dim, RenderContext* pRenderContext);

    void runTest(RenderContext* pRenderContext);

    void computeEvaluation(RenderContext* pRenderContext, SVGFRenderData& svgfrd, bool updateInternalBuffers);
    void computeBackPropagation(RenderContext* pRenderContext, SVGFRenderData& svgfrd);
private:
    ref<Device> mpDevice;
    ref<SVGFUtilitySet> mpUtilities;
    ref<FilterParameterReflector> mpParameterReflector;

    int mFilterIterations = 4;
    int mFeedbackTap = -1;

    ref<ComputePass> mpEvaluatePass;
    ref<ComputePass> mpBackPropagatePass;

    ref<Texture> mpTestIllum;
    ref<Texture> mpTestNormalDepth;
    ref<Texture> mpTestOutput;




    struct ConvolutionKernel
    {
        // map, y first, x second
        union
        {
            float weights[kOutputMapsPerLayer][kKernelDim][kKernelDim];
            float4 packed_weights[(kOutputMapsPerLayer * kKernelDim * kKernelDim + 3) / 4];
        };
        float bias;

        float fetch_weight(const int map, const int x, const int y);
    };

    struct PostconvolutionKernel
    {
        union
        {
            float weights[5][5];
            float4 packed_weights[(kMapDim * kMapDim + 3) / 4];
        };

        float fetch_weight(const int x, const int y);
    };

    struct ConvolutionMap
    {
        // indexing: first y, then x
        float m[5][5];

        float& get(const int x, const int y);
    };

    PostconvolutionKernel mPostconvKernels[8];
    ConvolutionKernel mKernels[32];

    float4 mTestIllumData[5][5];
    float4 mTestNormalData[5][5];
    ConvolutionMap mRbuf[kRingBufferSize];

    void print_test_result(float4 grid[][5]);
    void simulate_thread_group_sequentially(std::function<void(uint2)> func);
    void simulate_kpcnn();
    void convolve_kernel(uint2 srcPix, int readIdx, int writeIdx, int kernelIdx);
    void reduce_and_activate(uint2 offset, int writeIdx, int kernelIdx);
};

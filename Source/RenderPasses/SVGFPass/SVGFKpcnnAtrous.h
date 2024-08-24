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

    float4 mpTestIllumData[5][5];
    float4 mpTestNormalData[5][5];

    struct PostconvolutionKernel
    {
        union
        {
            float weights[5][5];
            float4 vwght[(kMapDim * kMapDim + 3) / 4];
        };
    };

    PostconvolutionKernel mpPostconvKernels[8];
};

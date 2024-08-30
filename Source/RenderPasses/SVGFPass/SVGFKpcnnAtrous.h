#pragma once

#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Core/Pass/FullScreenPass.h"
#include "Utils/Algorithm/ParallelReduction.h"
#include "Utils/Debug/PixelDebug.h"

#include "SVGFCommon.h"

using namespace Falcor;

#define kMapDim 5
#define kNumPixels (kMapDim * kMapDim)
#define kKernelDistance 1
#define kKernelDim 3
#define kKernelSummationTerms (kKernelDim * kKernelDim)
#define kOutputMapsPerLayer 8
#define kRingBufferDebugAdditionalSize 0
#define kRingBufferSize \
    (2 * kOutputMapsPerLayer + kKernelSummationTerms - 1 + kRingBufferDebugAdditionalSize) // minus one since for the last write index, we
                                                                                           // can simultaineously store/accum
#define kNumLayers 4
#define kNumOutputWeights kOutputMapsPerLayer
#define getRingBufferIndex(x) ((x) % kRingBufferSize)
#define arbuf(x) mRbuf[getRingBufferIndex(x)]
#define GET_RAW_WEIGHTS

class SVGFKpcnnAtrousSubpass : public Object
{
public:
    SVGFKpcnnAtrousSubpass(ref<Device> pDevice, ref<SVGFUtilitySet> pUtilities, ref<FilterParameterReflector> pParameterReflector);
    void allocateFbos(uint2 dim, RenderContext* pRenderContext);

    void runTest(RenderContext* pRenderContext);

    void computeEvaluation(RenderContext* pRenderContext, SVGFRenderData& svgfrd, bool updateInternalBuffers);
    void computeBackPropagation(RenderContext* pRenderContext, SVGFRenderData& svgfrd);

    void renderUI(Gui::Widgets& widget);

private:
    ref<Device> mpDevice;
    ref<SVGFUtilitySet> mpUtilities;
    ref<FilterParameterReflector> mpParameterReflector;

    int mFilterIterations = 4;
    int mFeedbackTap = -1;

    ref<ComputePass> mpEvaluatePass;
    ref<ComputePass> mpBackPropagatePass;

    std::unique_ptr<PixelDebug> mpPixelDebug;
    

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
        float padding[3];

        float fetch_weight(const int map, const int x, const int y);
    };

    struct PostconvolutionKernel
    {
        union
        {
            float weights[kMapDim][kMapDim];
            float4 packed_weights[(kMapDim * kMapDim + 3) / 4];
        };

        float fetch_weight(const int x, const int y);
    };

    struct ConvolutionMap
    {
        // indexing: first y, then x
        float m[kMapDim][kMapDim];

        float& get(const int x, const int y);
    };

    uint2 mCurrentCpuDebugPixel = uint2(0, 0);
    bool mCpuPrintingEnabled = false;
    void printPixelDebug(const std::string& s, float x);

    PostconvolutionKernel mPostconvKernels[kNumOutputWeights];
    ConvolutionKernel mKernels[kOutputMapsPerLayer * kNumLayers];

    float4 mTestIllumData[kMapDim][kMapDim];
    float4 mTestNormalData[kMapDim][kMapDim];
    ConvolutionMap mRbuf[kRingBufferSize];

    void print_test_result(float4 grid[][kMapDim]);
    void simulate_thread_group_sequentially(std::function<void(uint2)> func);
    void simulate_kpcnn();
    void reference_convolution(int readIdx, int kernelIdx, ConvolutionMap tRbuf[]);
    void clear_accumulation_area(uint2 srcPix, int writeIdx);
    void convolve_kernel(uint2 srcPix, int readIdx, int writeIdx, int kernelIdx);
    void reduce_and_activate(uint2 offset, int writeIdx, int kernelIdx);
};

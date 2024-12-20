#pragma once

#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Core/Pass/FullScreenPass.h"
#include "Utils/Algorithm/ParallelReduction.h"
#include "Utils/Debug/PixelDebug.h"

#include "SVGFCommon.h"

using namespace Falcor;

#define kMapDim 5
#define kTransformerItems 25
#define kNumFeatures 8

class SVGFTransformer : public Object
{
public:
    SVGFTransformer(ref<Device> pDevice, ref<SVGFUtilitySet> pUtilities, ref<FilterParameterReflector> pParameterReflector);
    void allocateFbos(uint2 dim, RenderContext* pRenderContext);

    void computeEvaluation(RenderContext* pRenderContext, SVGFRenderData& svgfrd, bool updateInternalBuffers);
    void computeBackPropagation(RenderContext* pRenderContext, SVGFRenderData& svgfrd);

    void renderUI(Gui::Widgets& widget);

private:
    ref<Device> mpDevice;
    ref<SVGFUtilitySet> mpUtilities;
    ref<FilterParameterReflector> mpParameterReflector;

public:
    ref<Texture> mpTestIllum;
    ref<Texture> mpTestNormalDepth;
    ref<Texture> mpTestOutput;
    ref<Texture> mpDebugBuf;

private:

    std::unique_ptr<PixelDebug> mpPixelDebug;

    struct WeightMatrix
    {
        union
        {
            float4 packed_weights[(kNumFeatures * kNumFeatures - 1) / 4 + 1];
            float weights[kNumFeatures][kNumFeatures];
        };
    };

    SVGFParameter<WeightMatrix[3]> mWeights;

    int mFilterIterations = 4;

    ref<ComputePass> mpEvaluatePass;
    ref<ComputePass> mpBackPropagatePass;

    ref<Texture> mpOutput;

    float4 mTestIllumData[kMapDim][kMapDim * 2];
    float4 mTestNormalData[kMapDim][kMapDim * 2];

    void set_common_parameters(ShaderVar& perImageCB, SVGFRenderData& svgfrd);
    void set_and_update_test_data(RenderContext* pRenderContext);
    void print_test_result(float4 grid[][kMapDim]);
};

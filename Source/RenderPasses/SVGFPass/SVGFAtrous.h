#pragma once

#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Core/Pass/FullScreenPass.h"
#include "Utils/Algorithm/ParallelReduction.h"

#include "SVGFCommon.h"

using namespace Falcor;

class SVGFAtrousSubpass : public Object
{
public:
    SVGFAtrousSubpass(ref<Device> pDevice, ref<SVGFUtilitySet> pUtilities, ref<FilterParameterReflector> pParameterReflector);
    void allocateFbos(uint2 dim, RenderContext* pRenderContext);

    void computeEvaluation(RenderContext* pRenderContext, SVGFRenderData& svgfrd, bool updateInternalBuffers);
    void computeBackPropagation(RenderContext* pRenderContext, SVGFRenderData& svgfrd);
private:
    ref<Device> mpDevice;
    ref<SVGFUtilitySet> mpUtilities;
    ref<FilterParameterReflector> mpParameterReflector;

    int mFilterIterations = 4;
    int mFeedbackTap = 1;

    ref<FullScreenPass> mpEvaluatePass;
    ref<FullScreenPass> mpBackPropagatePass;

    ref<Fbo> mpPingPongFbo[2];

public: // public so we can do derivative stuff
    struct PerIterationState
    {
        ref<Texture> pgIllumination; // saved illumination for this iteration

        SVGFParameter<float3> mSigma;

        SVGFParameter<float[3]> mWeightFunctionParams;
        SVGFParameter<float3> mLuminanceParams;

        SVGFParameter<float[2][2]> mVarianceKernel;
        SVGFParameter<float[3]> mKernel;
    };

    std::vector<PerIterationState> mIterationState;
};

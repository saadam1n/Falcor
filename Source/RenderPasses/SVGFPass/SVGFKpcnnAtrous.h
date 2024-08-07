#pragma once

#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Core/Pass/FullScreenPass.h"
#include "Utils/Algorithm/ParallelReduction.h"

#include "SVGFCommon.h"

using namespace Falcor;

class SVGFKpcnnAtrousSubpass : public Object
{
public:
    SVGFKpcnnAtrousSubpass(ref<Device> pDevice, ref<SVGFUtilitySet> pUtilities, ref<FilterParameterReflector> pParameterReflector);
    void allocateFbos(uint2 dim, RenderContext* pRenderContext);

    void computeEvaluation(RenderContext* pRenderContext, SVGFRenderData& svgfrd, bool updateInternalBuffers);
    void computeBackPropagation(RenderContext* pRenderContext, SVGFRenderData& svgfrd);
private:
    ref<Device> mpDevice;
    ref<SVGFUtilitySet> mpUtilities;
    ref<FilterParameterReflector> mpParameterReflector;

    int mFilterIterations = 4;
    int mFeedbackTap = -1;

    ref<FullScreenPass> mpEvaluatePass;
    ref<FullScreenPass> mpBackPropagatePass;


};

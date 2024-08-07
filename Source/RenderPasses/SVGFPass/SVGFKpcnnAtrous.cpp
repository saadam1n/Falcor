#include "SVGFKpcnnAtrous.h"

SVGFKpcnnAtrousSubpass::SVGFKpcnnAtrousSubpass(ref<Device> pDevice, ref<SVGFUtilitySet> pUtilities, ref<FilterParameterReflector> pParameterReflector)
    : mpDevice(pDevice), mpUtilities(pUtilities), mpParameterReflector(pParameterReflector)
{
    mpEvaluatePass = ComputePass::create(mpDevice, kKpcnnAtrousShaderS);
    mpBackPropagatePass = ComputePass::create(mpDevice, kKpcnnAtrousShaderD);
}

void SVGFKpcnnAtrousSubpass::allocateFbos(uint2 dim, RenderContext* pRenderContext)
{
}

void SVGFKpcnnAtrousSubpass::computeEvaluation(RenderContext* pRenderContext, SVGFRenderData& svgfrd, bool updateInternalBuffers)
{
}

void SVGFKpcnnAtrousSubpass::computeBackPropagation(RenderContext* pRenderContext, SVGFRenderData& svgfrd)
{
}

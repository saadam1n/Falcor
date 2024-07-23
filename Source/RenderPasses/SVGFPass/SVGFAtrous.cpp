#include "SVGFAtrous.h"

SVGFAtrousSubpass::SVGFAtrousSubpass(ref<Device> pDevice, ref<SVGFUtilitySet> pUtilities, ref<FilterParameterReflector> pParameterReflector) : mpDevice(pDevice), mpUtilities(pUtilities), mpParameterReflector(pParameterReflector)
{
    mpEvaluatePass = mpUtilities->createFullscreenPassAndDumpIR(kAtrousShaderS);
    mpBackPropagatePass = mpUtilities->createFullscreenPassAndDumpIR(kAtrousShaderD);

    // Set atrous state vars
    mIterationState.resize(mFilterIterations);
    for (auto& iterationState : mIterationState)
    {
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                iterationState.mSigmaL.dv[i][j] = dvSigma.x;
                iterationState.mSigmaZ.dv[i][j] = dvSigma.y;
                iterationState.mSigmaN.dv[i][j] = dvSigma.z;
            }
        }

        REGISTER_PARAMETER(mpParameterReflector, iterationState.mSigmaL);
        REGISTER_PARAMETER(mpParameterReflector, iterationState.mSigmaZ);
        REGISTER_PARAMETER(mpParameterReflector, iterationState.mSigmaN);

        for (int i = 0; i < 3; i++) {
            iterationState.mWeightFunctionParams.dv[i] = dvWeightFunctionParams[i];
        }
        REGISTER_PARAMETER(mpParameterReflector, iterationState.mWeightFunctionParams);

        iterationState.mLuminanceParams.dv = dvLuminanceParams;
        REGISTER_PARAMETER(mpParameterReflector, iterationState.mLuminanceParams);

        float defaultKernel[] = {1.0f, 2.0f / 3.0f, 1.0f / 6.0f};
        for (int yy = -2; yy <= 2; yy++)
        {
            for (int xx = -2; xx <= 2; xx++)
            {
                iterationState.mKernel.dv[yy + 2][xx + 2] = defaultKernel[abs(xx)] * defaultKernel[abs(yy)];
            }
        }

        REGISTER_PARAMETER(mpParameterReflector, iterationState.mKernel);

        iterationState.mVarianceKernel.dv[0][0] = 1.0 / 4.0;
        iterationState.mVarianceKernel.dv[0][1] = 1.0 / 8.0;
        iterationState.mVarianceKernel.dv[1][0] = 1.0 / 8.0;
        iterationState.mVarianceKernel.dv[1][1] = 1.0 / 16.0;
        REGISTER_PARAMETER(mpParameterReflector, iterationState.mVarianceKernel);
    }
}

void SVGFAtrousSubpass::allocateFbos(uint2 dim, RenderContext* pRenderContext)
{
    {
        // Screen-size FBOs with 1 RGBA32F buffer
        Fbo::Desc desc;
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float);

        mpPingPongFbo[0]  = Fbo::create2D(mpDevice, dim.x, dim.y, desc);
        mpPingPongFbo[1]  = Fbo::create2D(mpDevice, dim.x, dim.y, desc);
    }
}

void SVGFAtrousSubpass::computeEvaluation(RenderContext* pRenderContext, SVGFRenderData& svgfrd, bool updateInternalBuffers)
{
    FALCOR_PROFILE(pRenderContext, "Atrous");

    auto perImageCB = mpEvaluatePass->getRootVar()["PerImageCB"];

    perImageCB["gAlbedo"] = svgfrd.pAlbedoTexture;
    perImageCB["gLinearZAndNormal"] = svgfrd.fetchTexTable("gLinearZAndNormal");

    for (int iteration = 0; iteration < mFilterIterations; iteration++)
    {
        FALCOR_PROFILE(pRenderContext, "Iteration" + std::to_string(iteration));

        auto& curIterationState = mIterationState[iteration];

        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                perImageCB["dvSigmaL"][i][j] = curIterationState.mSigmaL.dv[i][j];
                perImageCB["dvSigmaZ"][i][j] = curIterationState.mSigmaZ.dv[i][j];
                perImageCB["dvSigmaN"][i][j] = curIterationState.mSigmaN.dv[i][j];
            }
        }

        perImageCB["dvLuminanceParams"] = curIterationState.mLuminanceParams.dv;

        for (int i = 0; i < 3; i++) {
            perImageCB["dvWeightFunctionParams"][i] = curIterationState.mWeightFunctionParams.dv[i];
        }

        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++)
            {
                perImageCB["dvKernel"][i][j] = curIterationState.mKernel.dv[i][j];
            }
        }

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                perImageCB["dvVarianceKernel"][i][j] = curIterationState.mVarianceKernel.dv[i][j];
            }
        }


        ref<Fbo> curTargetFbo = mpPingPongFbo[1];
        // keep a copy of our input for backwards differation
        svgfrd.saveInternalTex(pRenderContext, "AtrousIllum" + std::to_string(iteration), mpPingPongFbo[0]->getColorTexture(0));

        perImageCB["gIllumination"] = (iteration == 0 ? svgfrd.fetchTexTable("AtrousInputIllumination") : mpPingPongFbo[0]->getColorTexture(0));
        perImageCB["gStepSize"] = 1 << iteration;


        mpEvaluatePass->execute(pRenderContext, curTargetFbo);

        // store the filtered color for the feedback path
        if (updateInternalBuffers && iteration == std::min(mFeedbackTap, mFilterIterations - 1))
        {
            pRenderContext->blit(curTargetFbo->getColorTexture(0)->getSRV(), svgfrd.fetchTexTable("FilteredPast")->getRTV());
        }

        std::swap(mpPingPongFbo[0], mpPingPongFbo[1]);

        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                //std::cout << curIterationState.mSigmaL.dv[i][j] << "\n";
                //std::cout << curIterationState.mSigmaZ.dv[i][j] << "\n";
                //std::cout << curIterationState.mSigmaN.dv[i][j] << "\n";
            }
        }
    }

    if (updateInternalBuffers && mFeedbackTap < 0)
    {
        pRenderContext->blit(svgfrd.fetchTexTable("ReprojOutputCurIllum")->getSRV(), svgfrd.fetchTexTable("FilteredPast")->getRTV());
    }

    svgfrd.fetchTexTable("FinalModulateInIllumination") = mpPingPongFbo[0]->getColorTexture(0);


}

void SVGFAtrousSubpass::computeBackPropagation(RenderContext* pRenderContext, SVGFRenderData& svgfrd)
{
    FALCOR_PROFILE(pRenderContext, "Bwd Atrous");

    mpUtilities->setPatchingState(mpBackPropagatePass);

    auto perImageCB = mpBackPropagatePass->getRootVar()["PerImageCB"];
    auto perImageCB_D = mpBackPropagatePass->getRootVar()["PerImageCB_D"];

    perImageCB["gAlbedo"]           = svgfrd.pAlbedoTexture;
    perImageCB["gLinearZAndNormal"] = svgfrd.fetchTexTable("gLinearZAndNormal");

    for (int iteration = mFilterIterations - 1; iteration >= 0; iteration--)
    {
        FALCOR_PROFILE(pRenderContext, "Iteration" + std::to_string(iteration));

        auto& curIterationState = mIterationState[iteration];

        // clear raw output
        mpUtilities->clearRawOutputBuffer(pRenderContext, 0);

        perImageCB["daSigmaL"] = curIterationState.mSigmaL.da;
        perImageCB["daSigmaZ"] = curIterationState.mSigmaZ.da;
        perImageCB["daSigmaN"] = curIterationState.mSigmaN.da;
        perImageCB["daKernel"] = curIterationState.mKernel.da;
        perImageCB_D["daVarianceKernel"] = curIterationState.mVarianceKernel.da;
        perImageCB_D["daLuminanceParams"] = curIterationState.mLuminanceParams.da;
        perImageCB_D["daWeightFunctionParams"] = curIterationState.mWeightFunctionParams.da;

        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                perImageCB["dvSigmaL"][i][j] = curIterationState.mSigmaL.dv[i][j];
                perImageCB["dvSigmaZ"][i][j] = curIterationState.mSigmaZ.dv[i][j];
                perImageCB["dvSigmaN"][i][j] = curIterationState.mSigmaN.dv[i][j];
            }
        }

        perImageCB["dvLuminanceParams"] = curIterationState.mLuminanceParams.dv;

        for (int i = 0; i < 3; i++) {
            perImageCB["dvWeightFunctionParams"][i] = curIterationState.mWeightFunctionParams.dv[i];
        }

        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++)
            {
                perImageCB["dvKernel"][i][j] = curIterationState.mKernel.dv[i][j];
            }
        }

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                perImageCB["dvVarianceKernel"][i][j] = curIterationState.mVarianceKernel.dv[i][j];
            }
        }

        perImageCB_D["drIllumination"] = (iteration == mFilterIterations - 1 ? svgfrd.fetchBufTable("AtrousInIllumination") : mpUtilities->mpdrCompactedBuffer[0]);
        perImageCB["daIllumination"] = mpUtilities->mpdaRawOutputBuffer[0];

        perImageCB["gIllumination"] = svgfrd.fetchInternalTex("AtrousIllum" + std::to_string(iteration));
        perImageCB["gStepSize"] = 1 << iteration;

        mpBackPropagatePass->execute(pRenderContext, mpUtilities->getDummyFullscreenFbo());

        mpUtilities->runCompactingPass(pRenderContext, 0, 9 + 25);
    }

    svgfrd.fetchBufTable("FilterMomentsInIllumination") = mpUtilities->mpdrCompactedBuffer[0];
}

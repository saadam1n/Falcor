/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "SVGFPass.h"
#include <fstream>
#include <random>

/*
TODO:
- clean up shaders
- clean up UI: tooltips, etc.
- handle skybox pixels
- enum for fbo channel indices
*/

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, SVGFPass>();
}

SVGFPass::SVGFPass(ref<Device> pDevice, const Properties& props) :
    RenderPass(pDevice),
    mpUtilities(make_ref<SVGFUtilitySet>(pDevice, 200, 200, 800, 800)),
    mpParameterReflector(make_ref<FilterParameterReflector>(mpUtilities)),
    mRenderData(pDevice, mpUtilities),
    mTrainingDataset(pDevice, mpUtilities, "C:/FalcorFiles/Dataset0/")
{
    for (const auto& [key, value] : props)
    {
        if (key == kEnabled) mFilterEnabled = value;
        else if (key == kIterations) mFilterIterations = value;
        else if (key == kFeedbackTap) mFeedbackTap = value;
        else if (key == kVarianceEpsilon) mVarainceEpsilon = value;
        else if (key == kPhiColor) ;//dvSigmaL = value;
        else if (key == kPhiNormal) ;//dvSigmaN = value;
        else if (key == kAlpha) ;//dvAlpha = value;
        else if (key == kMomentsAlpha) ;//dvMomentsAlpha = value;
        else logWarning("Unknown field '{}' in SVGFPass dictionary.", key);
    }

    mPackLinearZAndNormalState.sPass = mpUtilities->createFullscreenPassAndDumpIR(kPackLinearZAndNormalShader);

    mReprojectState.sPass = mpUtilities->createFullscreenPassAndDumpIR(kReprojectShaderS);
    mReprojectState.dPass = mpUtilities->createFullscreenPassAndDumpIR(kReprojectShaderD);

    mFilterMomentsState.sPass = mpUtilities->createFullscreenPassAndDumpIR(kFilterMomentShaderS);
    mFilterMomentsState.dPass = mpUtilities->createFullscreenPassAndDumpIR(kFilterMomentShaderD);



    mFinalModulateState.sPass = mpUtilities->createFullscreenPassAndDumpIR(kFinalModulateShaderS);
    mFinalModulateState.dPass = mpUtilities->createFullscreenPassAndDumpIR(kFinalModulateShaderD);

    summingPass = ComputePass::create(mpDevice, kBufferShaderSumming);
    bufferToTexturePass = mpUtilities->createFullscreenPassAndDumpIR(kBufferShaderToTexture);

    mLossState.dPass = mpUtilities->createFullscreenPassAndDumpIR(kLossShader);
    mLossState.sGaussianPass = mpUtilities->createFullscreenPassAndDumpIR(kLossGaussianShaderS);
    mLossState.dGaussianPass = mpUtilities->createFullscreenPassAndDumpIR(kLossGaussianShaderD);

    mpDerivativeVerify = mpUtilities->createFullscreenPassAndDumpIR(kDerivativeVerifyShader);
    mpFuncOutputLower =  make_ref<Texture>(pDevice, Resource::Type::Texture2D, ResourceFormat::RGBA32Float, screenWidth, screenHeight,  1, 1, 1, 1, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource, nullptr);
    mpFuncOutputUpper =  make_ref<Texture>(pDevice, Resource::Type::Texture2D, ResourceFormat::RGBA32Float, screenWidth, screenHeight,  1, 1, 1, 1, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource, nullptr);

    // set linear z params



    // set reproj params
    mReprojectState.pdaIllumination = mpUtilities->createAccumulationBuffer();
    mReprojectState.pdaMoments = mpUtilities->createAccumulationBuffer();
    mReprojectState.pdaHistoryLength = mpUtilities->createAccumulationBuffer();

    mReprojectState.mLuminanceParams.dv = dvLuminanceParams;
    REGISTER_PARAMETER(mpParameterReflector, mReprojectState.mLuminanceParams);

    mReprojectState.mAlpha.dv = dvAlpha;
    REGISTER_PARAMETER(mpParameterReflector, mReprojectState.mAlpha);

    mReprojectState.mMomentsAlpha.dv = dvMomentsAlpha;
    REGISTER_PARAMETER(mpParameterReflector, mReprojectState.mMomentsAlpha);

    mReprojectState.mParams.dv[0] = 32.0;
    mReprojectState.mParams.dv[1] = 1.0;
    mReprojectState.mParams.dv[2] = 10.0;
    mReprojectState.mParams.dv[3] = 16.0;
    REGISTER_PARAMETER(mpParameterReflector, mReprojectState.mParams);

    mReprojectState.mKernel.dv[0] = 1.0;
    mReprojectState.mKernel.dv[1] = 1.0;
    mReprojectState.mKernel.dv[2] = 1.0;
    REGISTER_PARAMETER(mpParameterReflector, mReprojectState.mKernel);


    std::mt19937 mlp_rng(1234567);
    std::uniform_real_distribution<> mlp_offset(0.0f, 0.1f);
    for(int i = 0; i < kNumReprojectionMlpWeights; i++)
    {
        mReprojectState.mTemporalMlpWeights.dv[i] = mlp_offset(mlp_rng);
    }
    REGISTER_PARAMETER(mpParameterReflector, mReprojectState.mTemporalMlpWeights);




    // set filter moments params
    mFilterMomentsState.pdaHistoryLen = mpUtilities->createAccumulationBuffer();

    mFilterMomentsState.mSigma.dv = dvSigma;
    REGISTER_PARAMETER(mpParameterReflector, mFilterMomentsState.mSigma);

    mFilterMomentsState.mLuminanceParams.dv = dvLuminanceParams;
    REGISTER_PARAMETER(mpParameterReflector, mFilterMomentsState.mLuminanceParams);

    for (int i = 0; i < 3; i++) {
        mFilterMomentsState.mWeightFunctionParams.dv[i] = dvWeightFunctionParams[i];
    }
    REGISTER_PARAMETER(mpParameterReflector, mFilterMomentsState.mWeightFunctionParams);

    mFilterMomentsState.mVarianceBoostFactor.dv = 4.0;
    REGISTER_PARAMETER(mpParameterReflector, mFilterMomentsState.mVarianceBoostFactor);


    // set atrous state
    mpAtrousSubpass = make_ref<SVGFAtrousSubpass>(mpDevice, mpUtilities, mpParameterReflector);
    mpKpcnnAtrousSubpass = make_ref<SVGFKpcnnAtrousSubpass>(mpDevice, mpUtilities, mpParameterReflector);

    // set final modulate state vars
    mFinalModulateState.pdaIllumination = mpUtilities->createAccumulationBuffer();

    // set loss vars
    mLossState.pGaussianXInput = mpUtilities->createFullscreenTexture();
    mLossState.pGaussianYInput = mpUtilities->createFullscreenTexture();
    mLossState.pFilteredGaussian = mpUtilities->createFullscreenTexture();
    mLossState.pReferenceGaussian = mpUtilities->createFullscreenTexture();

    for (int i = 0; i < 2; i++)
    {
        pdaPingPongSumBuffer[i] = mpUtilities->createAccumulationBuffer(sizeof(float4));
    }

    for (int i = 0; i < 3; i++)
    {
        mReadbackBuffer[i] = mpUtilities->createAccumulationBuffer(sizeof(float4), true);
    }

    mpParallelReduction = std::make_unique<ParallelReduction>(mpDevice);
}

void SVGFPass::clearBuffers(RenderContext* pRenderContext, const SVGFRenderData& renderData)
{
    pRenderContext->clearFbo(mpPingPongFbo[0].get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpPingPongFbo[1].get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpLinearZAndNormalFbo.get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpFilteredPastFbo.get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpCurReprojFbo.get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpPrevReprojFbo.get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpFilteredIlluminationFbo.get(), float4(0), 1.0f, 0, FboAttachmentType::All);

    pRenderContext->clearTexture(renderData.pPrevLinearZAndNormalTexture.get());

    pRenderContext->clearFbo(mpDerivativeVerifyFbo.get(), float4(0), 1.0f, 0, FboAttachmentType::All);

    pRenderContext->clearTexture(renderData.pPrevFiltered.get());
    pRenderContext->clearTexture(renderData.pPrevReference.get());
}

void SVGFPass::allocateFbos(uint2 dim, RenderContext* pRenderContext)
{
    {
        // Screen-size FBOs with 3 MRTs: one that is RGBA32F, one that is
        // RG32F for the luminance moments, and one that is R16F.
        Fbo::Desc desc;
        desc.setSampleCount(0);
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float); // illumination
        desc.setColorTarget(1, Falcor::ResourceFormat::RG32Float); // moments
        desc.setColorTarget(2, Falcor::ResourceFormat::R32Float); // history length
        desc.setColorTarget(3, Falcor::ResourceFormat::RGBA32Float); // temporal accum
        mpCurReprojFbo  = Fbo::create2D(mpDevice, dim.x, dim.y, desc);
        mpPrevReprojFbo = Fbo::create2D(mpDevice, dim.x, dim.y, desc);
    }

    {
        // Screen-size RGBA32F buffer for linear Z, derivative, and packed normal
        Fbo::Desc desc;
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float);
        mpLinearZAndNormalFbo = Fbo::create2D(mpDevice, dim.x, dim.y, desc);
    }

    {
        // Screen-size FBOs with 1 RGBA32F buffer
        Fbo::Desc desc;
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float);

        mpPingPongFbo[0]  = Fbo::create2D(mpDevice, dim.x, dim.y, desc);
        mpPingPongFbo[1]  = Fbo::create2D(mpDevice, dim.x, dim.y, desc);
        mpFilteredIlluminationFbo = Fbo::create2D(mpDevice, dim.x, dim.y, desc);
        mpFinalFbo = Fbo::create2D(mpDevice, dim.x, dim.y, desc);
    }

    {
        // One buffer for each of the atrous passes
        Fbo::Desc desc;
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float);
        desc.setColorTarget(1, Falcor::ResourceFormat::RGBA32Float);
        desc.setColorTarget(2, Falcor::ResourceFormat::RGBA32Float);
        desc.setColorTarget(3, Falcor::ResourceFormat::RGBA32Float);
        desc.setColorTarget(4, Falcor::ResourceFormat::RGBA32Float); //4th one for the filter moments pass

        mpFilteredPastFbo = Fbo::create2D(mpDevice, dim.x, dim.y, desc);
    }

    {
        Fbo::Desc desc;
        desc.setSampleCount(0);
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float);
        desc.setColorTarget(1, Falcor::ResourceFormat::RGBA32Float);
        desc.setColorTarget(2, Falcor::ResourceFormat::RGBA32Float);
        mpDerivativeVerifyFbo = Fbo::create2D(mpDevice, dim.x, dim.y, desc);

    }

    {
        Fbo::Desc desc;
        desc.setSampleCount(0);
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float);
        bufferToTextureFbo = Fbo::create2D(mpDevice, dim.x, dim.y, desc);
    }

    {
        // The gaussian pass
        Fbo::Desc desc;
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float);

        for(int i = 0; i < 2; i++)
        {
            mLossState.pGaussianFbo[i] = Fbo::create2D(mpDevice, dim.x, dim.y, desc);
        }

    }

    mpUtilities->allocateFbos(dim, pRenderContext);
    mpAtrousSubpass->allocateFbos(dim, pRenderContext);
    mpKpcnnAtrousSubpass->allocateFbos(dim, pRenderContext);

    mBuffersNeedClear = true;
}

Properties SVGFPass::getProperties() const
{
    Properties dict;
    dict[kEnabled] = mFilterEnabled;
    dict[kIterations] = mFilterIterations;
    dict[kFeedbackTap] = mFeedbackTap;
    dict[kVarianceEpsilon] = mVarainceEpsilon;
    // doesn't really make sense for our use case
    dict[kPhiColor] = -1.0;
    dict[kPhiNormal] = -1.0;
    dict[kAlpha] = -1.0;
    dict[kMomentsAlpha] = -1.0;
    return dict;
}

/*
Reproject:
  - takes: motion, color, prevLighting, prevMoments, linearZ, prevLinearZ, historyLen
    returns: illumination, moments, historyLength
Variance/filter moments:
  - takes: illumination, moments, history length, normal+depth
  - returns: filtered illumination+variance (to ping pong fbo)
a-trous:
  - takes: albedo, filtered illumination+variance, normal+depth, history length
  - returns: final color
*/

RenderPassReflection SVGFPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    reflector.addInput(kInputBufferAlbedo, "Albedo");
    reflector.addInput(kInputBufferColor, "Color");
    reflector.addInput(kInputBufferEmission, "Emission");
    reflector.addInput(kInputBufferWorldPosition, "World Position");
    reflector.addInput(kInputBufferWorldNormal, "World Normal");
    reflector.addInput(kInputBufferPosNormalFwidth, "PositionNormalFwidth");
    reflector.addInput(kInputBufferLinearZ, "LinearZ");
    reflector.addInput(kInputBufferMotionVector, "Motion vectors");

    reflector.addInternal(kInternalBufferPreviousLinearZAndNormal, "Previous Linear Z and Packed Normal")
        .format(ResourceFormat::RGBA32Float)
        .bindFlags(ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource);
    reflector.addInternal(kInternalBufferPreviousLighting, "Previous Filtered Lighting")
        .format(ResourceFormat::RGBA32Float)
        .bindFlags(ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource);
    reflector.addInternal(kInternalBufferPreviousMoments, "Previous Moments")
        .format(ResourceFormat::RG32Float)
        .bindFlags(ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource);

    reflector.addInternal(kInternalBufferPreviousFiltered, "Previous filtered")
        .format(ResourceFormat::RGBA32Float)
        .bindFlags(ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource);

    reflector.addInternal(kInternalBufferPreviousReference, "Previous reference")
        .format(ResourceFormat::RGBA32Float)
        .bindFlags(ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource);

    reflector.addOutput(kOutputBufferFilteredImage, "Filtered image").format(ResourceFormat::RGBA16Float);
    reflector.addOutput(kOutputDebugBuffer, "DebugBuf").format(ResourceFormat::RGBA16Float);
    reflector.addOutput(kOutputDerivVerifyBuf, "Deriv Verify").format(ResourceFormat::RGBA16Float);
    reflector.addOutput(kOutputFuncLower, "Func lower").format(ResourceFormat::RGBA16Float);
    reflector.addOutput(kOutputFuncUpper, "Func upper").format(ResourceFormat::RGBA16Float);
    reflector.addOutput(kOutputFdCol, "FdCol").format(ResourceFormat::RGBA16Float);
    reflector.addOutput(kOutputBdCol, "BdCol").format(ResourceFormat::RGBA16Float);
    reflector.addOutput(kOutputReference, "Reference").format(ResourceFormat::RGBA16Float);
    reflector.addOutput(kOutputLoss, "Loss").format(ResourceFormat::RGBA16Float);
    reflector.addOutput(kOutputCenterLoss, "CenterLoss").format(ResourceFormat::RGBA16Float);
    reflector.addOutput(kOutputGradientLoss, "GradientLoss").format(ResourceFormat::RGBA16Float);
    reflector.addOutput(kOutputTemporalLoss, "TemporalLoss").format(ResourceFormat::RGBA16Float);

    return reflector;
}

void SVGFPass::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    allocateFbos(compileData.defaultTexDims, pRenderContext);
    mBuffersNeedClear = true;
}

void SVGFPass::runSvgfFilter(RenderContext* pRenderContext, SVGFRenderData& renderData, bool updateInternalBuffers)
{
    FALCOR_PROFILE(pRenderContext, "SVGF Filter");

    if (mBuffersNeedClear)
    {
        clearBuffers(pRenderContext, renderData);
        mBuffersNeedClear = false;
    }

    if (mFilterEnabled)
    {
        // Grab linear z and its derivative and also pack the normal into
        // the last two channels of the mpLinearZAndNormalFbo.
        computeLinearZAndNormal(pRenderContext, renderData);

        // Demodulate input color & albedo to get illumination and lerp in
        // reprojected filtered illumination from the previous frame.
        // Stores the result as well as initial moments and an updated
        // per-pixel history length in mpCurReprojFbo.

        computeReprojection(pRenderContext, renderData);

        // Do a first cross-bilateral filtering of the illumination and
        // estimate its variance, storing the result into a float4 in
        // mpPingPongFbo[0].  Takes mpCurReprojFbo as input.
        computeFilteredMoments(pRenderContext, renderData);

        //pRenderContext->blit(mpPingPongFbo[0]->getColorTexture(1)->getSRV(), pDebugTexture->getRTV());

        // Filter illumination from mpCurReprojFbo[0], storing the result
        // in mpPingPongFbo[0].  Along the way (or at the end, depending on
        // the value of mFeedbackTap), save the filtered illumination for
        // next time into mpFilteredPastFbo.
        computeAtrousDecomposition(pRenderContext, renderData, updateInternalBuffers);

        // Compute albedo * filtered illumination and add emission back in.
        auto perImageCB = mFinalModulateState.sPass->getRootVar()["PerImageCB"];
        perImageCB["gAlbedo"] = renderData.pAlbedoTexture;
        perImageCB["gEmission"] = renderData.pEmissionTexture;
        perImageCB["gIllumination"] = renderData.fetchTexTable("FinalModulateInIllumination");
        mFinalModulateState.sPass->execute(pRenderContext, mpFinalFbo);
        renderData.saveInternalTex(pRenderContext, "FinalModulateFinalFiltered", mpPingPongFbo[0]->getColorTexture(0), false);

        if (updateInternalBuffers)
        {
            // Swap resources so we're ready for next frame.
            // only do it though if we are calculating derivaitves so we don't screw up our results from the finite diff pass
            std::swap(mpCurReprojFbo, mpPrevReprojFbo);
            pRenderContext->blit(mpLinearZAndNormalFbo->getColorTexture(0)->getSRV(),
                                    renderData.pPrevLinearZAndNormalTexture->getRTV());

        }

        // Blit into the output texture.
        pRenderContext->blit(mpFinalFbo->getColorTexture(0)->getSRV(), renderData.pOutputTexture->getRTV());
    }
    else
    {
        pRenderContext->blit(renderData.pColorTexture->getSRV(), renderData.pOutputTexture->getRTV());
    }
}

double getTexSum(RenderContext* pRenderContext, ref<Texture> tex)
{
    auto v = pRenderContext->readTextureSubresource(tex.get(), 0);

    float4* ptr = (float4*)v.data();

    double sum = 0.0;
    for(int i = 0; i < numPixels; i++)
        sum += (double)ptr[i].x;

    return sum;
}

void SVGFPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mKpcnnTested || mKeepRunningKpcnnTest)
    {
        mpKpcnnAtrousSubpass->runTest(pRenderContext);
        mKpcnnTested = true;
    }
    return;

    runTrainingAndTesting(pRenderContext, renderData);
    std::cout.flush();
}


const int K_NUM_EPOCHS = 1;
const int K_FRAME_SAMPLE_START = 48;

const float K_LRATE_NUMER = 25.0f * 0.0085f; // 0.0085 is a good value
const float K_LRATE_DENOM = 25.0f * 1.0f;

// parameters for the adam algorithm
// the paper recommends 0.9 and 0.999 respectively, but we observed that leads to exploding gradients
const float K_BETA_MOMENTUM = 0.9f;
const float K_BETA_SSGRAD = 0.999f;
const float K_ADAM_ELIPSON = 5e-3f;

void SVGFPass::runTrainingAndTesting(RenderContext* pRenderContext, const RenderData& renderData)
{
    mRenderData.copyTextureReferences(renderData);

    if (!mTrained)
    {
        runNextTrainingTask(pRenderContext);
        // display current results to screen
        pRenderContext->blit(mTrainingDataset.pOutputTexture->getSRV(), mRenderData.pOutputTexture->getRTV());
        pRenderContext->blit(mTrainingDataset.pReferenceTexture->getSRV(), mRenderData.pReferenceTexture->getRTV());
        pRenderContext->blit(mTrainingDataset.pLossTexture->getSRV(), mRenderData.pLossTexture->getRTV());
        pRenderContext->blit(mTrainingDataset.pCenterLossTexture->getSRV(), mRenderData.pCenterLossTexture->getRTV());
        pRenderContext->blit(mTrainingDataset.pGradientLossTexture->getSRV(), mRenderData.pGradientLossTexture->getRTV());
        pRenderContext->blit(mTrainingDataset.pTemporalLossTexture->getSRV(), mRenderData.pTemporalLossTexture->getRTV());
    }
    else
    {
        if(!pScene) return;

        runSvgfFilter(pRenderContext, mRenderData, true);

        // compute loss so we can see it on the screen
        pRenderContext->blit(mTrainingDataset.pReferenceTexture->getSRV(), mRenderData.pReferenceTexture->getRTV());
        computeLoss(pRenderContext, mRenderData, true); // keep it true (we aren't pushing the buffers so it wont matter)

        float4 loss;
        mpParallelReduction->execute(pRenderContext, mRenderData.pLossTexture, ParallelReduction::Type::Sum, &loss);

        // wait for all pending actions to execute
        pRenderContext->submit(true);
        std::cout << "Total loss: " << loss.x << "\n";
    }
}



static std::vector<float> lossHistory;
void SVGFPass::runNextTrainingTask(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "Next Training Task");

    mTrainingDataset.preloadBitmaps();
#if 0


    if(mEpoch < K_NUM_EPOCHS)
    {
        if(mEpoch == 0)
        {
            mBetaMomentumCorrection = K_BETA_MOMENTUM;
            mBetaSsgradCorrection = K_BETA_SSGRAD;
        }

        if (mDatasetIndex == 0) {
            mBuffersNeedClear = true;
            mReductionAddress = 0;
            std::cout << "Running epoch\t" << mEpoch << "\n";
        }

        if (!mBackPropagatingState && mTrainingDataset.loadNext(pRenderContext)) {
            if(mDatasetIndex % 10 == 0)
            {
                std::cout << "\tRendering Frame " << mDatasetIndex << "\n";
            }

            mTrainingDataset.setTimeframeState(false);
            runSvgfFilter(pRenderContext, mTrainingDataset, true);

            // add plus one so we save the previous frames resources
            if(mDatasetIndex + 1 >= K_FRAME_SAMPLE_START)
            {
#if 0
                saveLossBuffers(pRenderContext, mTrainingDataset);
                updateLossBuffers(pRenderContext, mTrainingDataset);
#else
                computeLoss(pRenderContext, mTrainingDataset, true);
                mpParallelReduction->execute<float4>(pRenderContext, mTrainingDataset.pLossTexture, ParallelReduction::Type::Sum, nullptr, mReadbackBuffer[2], (96 + mDatasetIndex) * sizeof(float4));
#endif


                mTrainingDataset.pushInternalBuffers(pRenderContext);

            }

            mDatasetIndex++;
            mFirstBackPropIteration = true;
        }
        else if(mDatasetIndex > K_FRAME_SAMPLE_START) // we have to use strictly greater for reasons
        {
            mBackPropagatingState = true;

            // first clear all our buffers
            clearTrainingBuffers(pRenderContext);

            if(mFirstBackPropIteration)
            {
                mBatchSize = mDatasetIndex;
                mFirstBackPropIteration = false;
            }

            mDatasetIndex--; // find what index this is associated with 

            if(mDatasetIndex % 10 == 0)
            {
                std::cout << "\tBackpropagating Frame " << mDatasetIndex << "\n";
            }



            mTrainingDataset.loadPrev(pRenderContext);
            mTrainingDataset.popInternalBuffers(pRenderContext);

            mTrainingDataset.setTimeframeState(true);
            runSvgfFilter(pRenderContext, mTrainingDataset, true);
            computeDerivatives(pRenderContext, mTrainingDataset, true);
            reduceAllData(pRenderContext);
        }
        else
        {
            int sampledFrames = mBatchSize - K_FRAME_SAMPLE_START;

            updateParameters(pRenderContext, sampledFrames);
            printLoss(pRenderContext, sampledFrames);

            mBetaMomentumCorrection *= K_BETA_MOMENTUM;
            mBetaSsgradCorrection *= K_BETA_SSGRAD;

            mBackPropagatingState = false;
            mReductionAddress = 0;
            mDatasetIndex = 0;
            mEpoch++;
            mTrainingDataset.reset();
        }
    }
    else
    {
        mTrained = true;
    }
#else

    if(mEpoch < K_NUM_EPOCHS)
    {
        if(mEpoch == 0)
        {
            mBetaMomentumCorrection = K_BETA_MOMENTUM;
            mBetaSsgradCorrection = K_BETA_SSGRAD;
        }

        if (mDatasetIndex == 0) {
            mBuffersNeedClear = true;
            mReductionAddress = 0;
            std::cout << "Running epoch\t" << mEpoch << "\n";
        }

        if (!mBackPropagatingState && mTrainingDataset.loadNext(pRenderContext)) {
            if(mDatasetIndex % 10 == 0)
            {
                std::cout << "\tRendering Frame " << mDatasetIndex << "\n";
            }
            // first clear all our buffers
            clearTrainingBuffers(pRenderContext);

            runSvgfFilter(pRenderContext, mTrainingDataset, true);
            //mTrainingDataset.pushInternalBuffers(pRenderContext);

            if(true || mDatasetIndex >= K_FRAME_SAMPLE_START)
            {
                //mTrainingDataset.popInternalBuffers(pRenderContext);
                //mTrainingDataset.saveInternalTex(pRenderContext, "LossOutput", mTrainingDataset.pOutputTexture, true);
                //mTrainingDataset.saveInternalTex(pRenderContext, "LossReference", mTrainingDataset.pReferenceTexture, true); // lazy way to do it tbh

                mTrainingDataset.saveInternalTex(pRenderContext, "LossPrevOutput", mTrainingDataset.pPrevFiltered, true);
                mTrainingDataset.saveInternalTex(pRenderContext, "LossPrevReference", mTrainingDataset.pPrevReference, true); // lazy way to do it tbh
                computeDerivatives(pRenderContext, mTrainingDataset, true);

                // now accumulate everything
                reduceAllData(pRenderContext);

                // keep a copy of our output
                //mTrainingDataset.pOutputTexture->captureToFile(0, 0, "C:/FalcorFiles/TrainingDump/" + std::to_string(epoch) + ".exr", Falcor::Bitmap::FileFormat::ExrFile, Falcor::Bitmap::ExportFlags::None, false);
            }

            mDatasetIndex++;
        }
        else if(mDatasetIndex >= K_FRAME_SAMPLE_START)
        {
            if(mDatasetIndex % 10 == 0)
            {
                std::cout << "\tBackpropagating Frame " << mDatasetIndex << "\n";
            }


            mBackPropagatingState = true;
            if(mFirstBackPropIteration)
            {
                mBatchSize = mDatasetIndex;
                mFirstBackPropIteration = false;
            }

            /*
            mTrainingDataset.loadPrev(pRenderContext);

            mTrainingDataset.popInternalBuffers(pRenderContext);
            runSvgfFilter(pRenderContext, mTrainingDataset, true);
            computeDerivatives(pRenderContext, mTrainingDataset, true);
            reduceAllData(pRenderContext);
            */

            mDatasetIndex--;
        }
        else
        {
            int sampledFrames = mBatchSize - K_FRAME_SAMPLE_START;

            updateParameters(pRenderContext, sampledFrames);
            printLoss(pRenderContext, sampledFrames);

            mBetaMomentumCorrection *= K_BETA_MOMENTUM;
            mBetaSsgradCorrection *= K_BETA_SSGRAD;

            mBackPropagatingState = false;
            mReductionAddress = 0;
            mDatasetIndex = 0;
            mEpoch++;
            mTrainingDataset.reset();
        }
    }
    else
    {
        mTrained = true;
    }
#endif
}

void SVGFPass::clearTrainingBuffers(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "Clr Param Buffers");
    for (auto& param : mpParameterReflector->mRegistry)
    {
        pRenderContext->clearUAV(param.mAccum->getUAV().get(), float4(0));

        if(mEpoch == 0)
        {
            for(int j = 0; j < param.mNumElements; j++)
            {
                param.momentum[j] = 0.0f;
                param.ssgrad[j] = 0.0f;
            }
        }
    }

    if (mFirstBackPropIteration)
    {
        pRenderContext->clearUAV(mReprojectState.pdaIllumination->getUAV().get(), float4(0.0f));
        pRenderContext->clearUAV(mReprojectState.pdaMoments->getUAV().get(), float4(0.0f));
        pRenderContext->clearUAV(mReprojectState.pdaHistoryLength->getUAV().get(), float4(0.0f));
    }
}

void SVGFPass::reduceAllData(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "Data Reduction");
    for (auto& param : mpParameterReflector->mRegistry)
    {
        mReductionAddress = reduceParameter(pRenderContext, param, mReductionAddress);
    }

    mpParallelReduction->execute<float4>(pRenderContext, mTrainingDataset.pLossTexture, ParallelReduction::Type::Sum, nullptr, mReadbackBuffer[2], mDatasetIndex * sizeof(float4));
}

std::ofstream alsoLog("C:\\FalcorFiles\\log.txt");

float SVGFPass::getAverageGradient(float* ptr, int baseOffset, int sampledFrames)
{
    float totalGradient = 0.0f;
    for (int k = 0; k < sampledFrames; k++)
    {
        alsoLog << "\t\t" << ptr[baseOffset + k * mpParameterReflector->getPackedStride()] << "\n";
        totalGradient += ptr[baseOffset + k * mpParameterReflector->getPackedStride()];
    }
    totalGradient /= sampledFrames;

    return totalGradient;
}

float SVGFPass::calculateBaseAdjustment(float gradient, float& momentum, float& ssgrad)
{
    float nextMomentum = K_BETA_MOMENTUM * momentum + (1.0f - K_BETA_MOMENTUM) * gradient;
    float nextSsgrad = K_BETA_SSGRAD * ssgrad + (1.0f - K_BETA_SSGRAD) * gradient * gradient;

    momentum = nextMomentum;
    ssgrad = nextSsgrad;

    float unbiasedMomentum = nextMomentum / (1.0f - mBetaMomentumCorrection);
    float unbiasedSsgrad = nextSsgrad / (1.0f - mBetaSsgradCorrection);

    return unbiasedMomentum / (sqrt(unbiasedSsgrad) + K_ADAM_ELIPSON);
}

void SVGFPass::updateParameters(RenderContext* pRenderContext, int sampledFrames)
{
    // skip the first few frames which probably don't have stablized derivatives

    float learningRate = K_LRATE_NUMER / (K_LRATE_DENOM + mEpoch);

    int currentOffset = 0;

    // adjust values
    float maxAdjValue = 0.0f;
    std::string maxAdjParamName = "none";
    std::vector<std::string> mismatchedParameters;

    float* gradient = (float*)mReadbackBuffer[0]->map();
    for (int i = 0; i < mpParameterReflector->getNumParams(); i++)
    {
        auto& pmi = mpParameterReflector->mRegistry[i];

        bool isMlpParameter = (pmi.mName.find("MlpWeights") != std::string::npos);

        for (int j = 0; j < pmi.mNumElements; j++)
        {
            float averageGradient = getAverageGradient(gradient, currentOffset, sampledFrames);

            float adjustment = learningRate * calculateBaseAdjustment(averageGradient, pmi.momentum[j], pmi.ssgrad[j]);

            //if(isMlpParameter) adjustment *= 25.0f;

            pmi.mAddress[j] -= adjustment;

            if(!isMlpParameter) pmi.mAddress[j] = std::max(pmi.mAddress[j], 0.0f); // gradient clipping

            currentOffset++;

            std::string printName = pmi.mName + (pmi.mNumElements != 1 ? "[" + std::to_string(j) + "]" : "");
            alsoLog << "\tAdjusting " << printName << "\tby " << -adjustment << "\twhen negative gradient is " << -averageGradient << "\n";
            std::cout << "\tAdjusting " << printName << "\tby " << -adjustment << "\twhen negative gradient is " << -averageGradient << "\n";
            if(sign(adjustment) != sign(averageGradient))
            {
                std::cout << "\tSign mismatch with " << averageGradient << "\n";
                mismatchedParameters.push_back(pmi.mName);
            }
            std::cout << "\n";

            if(abs(adjustment) > abs(maxAdjValue))
            {
                maxAdjValue = adjustment;
                maxAdjParamName = pmi.mName;
            }

        }

        // round up divide
        currentOffset = 4 * ((currentOffset + 3) / 4);
    }

    mReadbackBuffer[0]->unmap();


    std::cout << "Max adjustment was " << maxAdjValue << "\tfor " << maxAdjParamName  << "\n";

    std::cout << mismatchedParameters.size() << " mismatched parameters:\n";
    for(const auto& s : mismatchedParameters)
    {
        std::cout << "\t" << s << "\n";
    }

    alsoLog.flush();
    alsoLog.close();
}

void SVGFPass::printLoss(RenderContext* pRenderContext, int sampledFrames)
{
    // now wait for it to execute and download it
    float4 loss = float4(0.0f);
    float4 originalLoss = float4(0.0f);



    float4* perFrameLoss = (float4*)mReadbackBuffer[2]->map();
    for(int i = K_FRAME_SAMPLE_START; i < mBatchSize; i++)
    {
        loss += perFrameLoss[i];
        originalLoss += perFrameLoss[i + 96];

        std::cout << perFrameLoss[i].r / sampledFrames << "\t" << perFrameLoss[i + 96].r / sampledFrames << std::endl;
    }
    mReadbackBuffer[2]->unmap();



    loss /= float4(sampledFrames);
    originalLoss /= float4(sampledFrames);
    std::cout << "Average loss in epoch\t" << mEpoch << "\tacross " << sampledFrames << "\t frames was " << loss.r << "\n";
    std::cout << "Original loss in epoch\t" << mEpoch << "\tacross " << sampledFrames << "\t frames was " << originalLoss.r << "\n";

    lossHistory.push_back(loss.r);
    std::cout << "Loss history:\n";
    for(float l : lossHistory)
    {
        std::cout << "\t" << l << std::endl;
    }

    std::cout << "\n\n\n\n\n\n";
}


void SVGFPass::runDerivativeTest(RenderContext* pRenderContext, const RenderData& renderData)
{
    if(!pScene) return;

    mRenderData.copyTextureReferences(renderData);

    mDelta = 0.05f;

    float& valToChange = mpAtrousSubpass->mIterationState[mDerivativeIteration].mSigmaL.dv[2][2];
    float oldval = valToChange;

    valToChange = oldval - mDelta;
    runSvgfFilter(pRenderContext, mRenderData, false);
    pRenderContext->blit(mpFinalFbo->getColorTexture(0)->getSRV(), mpFuncOutputLower->getRTV());
    pRenderContext->blit(mpFinalFbo->getColorTexture(0)->getSRV(), renderData.getTexture(kOutputFuncLower)->getRTV());


    valToChange = oldval + mDelta;
    runSvgfFilter(pRenderContext, mRenderData, false);
    pRenderContext->blit(mpFinalFbo->getColorTexture(0)->getSRV(), mpFuncOutputUpper->getRTV());
    pRenderContext->blit(mpFinalFbo->getColorTexture(0)->getSRV(),  renderData.getTexture(kOutputFuncUpper)->getRTV());

    valToChange = oldval;

    runSvgfFilter(pRenderContext, mRenderData, true);
    computeDerivatives(pRenderContext, mRenderData, false);
    computeDerivVerification(pRenderContext, mRenderData);
    pRenderContext->blit(mpDerivativeVerifyFbo->getColorTexture(1)->getSRV(),  renderData.getTexture(kOutputFdCol)->getRTV());
    pRenderContext->blit(mpDerivativeVerifyFbo->getColorTexture(2)->getSRV(),  renderData.getTexture(kOutputBdCol)->getRTV());

    //pRenderContext->blit(mAtrousState.mSaveIllum->getSRV(),   renderData.getTexture(kOutputDebugBuffer)->getRTV());

    std::cout << "Fwd Diff Sum:\t" << getTexSum(pRenderContext, mpDerivativeVerifyFbo->getColorTexture(1)) << std::endl;
    std::cout << "Bwd Diff Sum:\t" << getTexSum(pRenderContext, mpDerivativeVerifyFbo->getColorTexture(2)) << std::endl;

    float4 falcorTest = float4(0.0f);

    mpParallelReduction->execute(pRenderContext, mpDerivativeVerifyFbo->getColorTexture(2), ParallelReduction::Type::Sum, &falcorTest);

    pRenderContext->submit(true);

    std::cout << "Flr Redc Sum:\t" << falcorTest.x << std::endl;

    std::cout << std::endl;
}

void SVGFPass::computeDerivVerification(RenderContext* pRenderContext, const SVGFRenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "Derivative Verif");

    auto perImageCB = mpDerivativeVerify->getRootVar()["PerImageCB"];

    perImageCB["drBackwardsDiffBuffer"] = mpAtrousSubpass->mIterationState[mDerivativeIteration].mSigmaL.da;
    perImageCB["gFuncOutputLower"] = mpFuncOutputLower;
    perImageCB["gFuncOutputUpper"] = mpFuncOutputUpper;
    perImageCB["delta"] = mDelta;

    mpDerivativeVerify->execute(pRenderContext, mpDerivativeVerifyFbo);
    pRenderContext->blit(mpDerivativeVerifyFbo->getColorTexture(0)->getSRV(), renderData.pDerivVerifyTexture->getRTV());
}

void SVGFPass::saveLossBuffers(RenderContext* pRenderContext, SVGFRenderData& renderData)
{
    //renderData.saveInternalTex(pRenderContext, "LossOutput", renderData.pOutputTexture, true);
    //renderData.saveInternalTex(pRenderContext, "LossReference", mTrainingDataset.pReferenceTexture, true); // lazy way to do it tbh


    renderData.saveInternalTex(pRenderContext, "LossPrevOutput", renderData.pPrevFiltered, true);
    renderData.saveInternalTex(pRenderContext, "LossPrevReference", renderData.pPrevReference, true); // lazy way to do it tbh
}

void SVGFPass::updateLossBuffers(RenderContext* pRenderContext, SVGFRenderData& renderData)
{
    // update the previous textures
    pRenderContext->blit(renderData.pOutputTexture->getSRV(), renderData.pPrevFiltered->getRTV());
    pRenderContext->blit(renderData.pOutputTexture->getSRV(), renderData.fetchInternalTex("LossPrevOutput")->getRTV());

    pRenderContext->blit(renderData.pReferenceTexture->getSRV(), renderData.pPrevReference->getRTV());
    pRenderContext->blit(renderData.pReferenceTexture->getSRV(), renderData.fetchInternalTex("LossPrevReference")->getRTV());
}


// I'll move parts of this off to other function as need be
void SVGFPass::computeDerivatives(RenderContext* pRenderContext, SVGFRenderData& renderData, bool useLoss)
{
    FALCOR_PROFILE(pRenderContext, "Bwd Pass");

    ref<Texture> pIllumTexture = mpPingPongFbo[0]->getColorTexture(0);


    if (mFilterEnabled) {
        if (useLoss)
        {
            computeLoss(pRenderContext, renderData, false);
        }
        else
        {
            // set everything to 1.0 (except the alpha channel)
            // we set everything to numepixels because the final modulate state divides by num pixels
            float4 defaultDerivative = float4(1.0, 1.0, 1.0, 0.0) * (float)numPixels;
            uint4* dPtr = (uint4*)&defaultDerivative;
            pRenderContext->clearUAV(mpUtilities->mpdrCompactedBuffer[1]->getUAV().get(), *dPtr);
        }

        computeDerivFinalModulate(pRenderContext, renderData);

        // now, the derivative is stored in mFinalModulateState.pdaIllum
        // now, we will have to computer the atrous decomposition reverse
        // the atrous decomp has multiple stages
        // each stage outputs the exact same result - a color
        // we need to use that color and its derivative to feed the previous pass
        // ideally, we will want a buffer and specific variables for each stage
        // right now, I'll just set up the buffers

        computeDerivAtrousDecomposition(pRenderContext, renderData);

        computeDerivFilteredMoments(pRenderContext, renderData);

        computeDerivReprojection(pRenderContext, renderData);
    }
}

void SVGFPass::computeLoss(RenderContext* pRenderContext, SVGFRenderData& renderData, bool saveInternalState)
{
    FALCOR_PROFILE(pRenderContext, "Loss");

    mpUtilities->setPatchingState(mLossState.dPass);

    if(saveInternalState)
    {
        saveLossBuffers(pRenderContext, renderData);
    }

    mpUtilities->executeDummyFullscreenPass(pRenderContext, renderData.pOutputTexture);
    mpUtilities->executeDummyFullscreenPass(pRenderContext, renderData.fetchInternalTex("LossOutput"));

    computeGaussian(pRenderContext, renderData.pReferenceTexture, mLossState.pReferenceGaussian, false);
    computeGaussian(pRenderContext, renderData.pOutputTexture, mLossState.pFilteredGaussian, true);

    mpUtilities->clearRawOutputBuffer(pRenderContext, 0);
    mpUtilities->clearRawOutputBuffer(pRenderContext, 1);

    auto perImageCB = mLossState.dPass->getRootVar()["PerImageCB"];

    perImageCB["filteredGaussian"] = mLossState.pFilteredGaussian;
    perImageCB["referenceGaussian"] = mLossState.pReferenceGaussian;

    perImageCB["filteredImage"] = renderData.pOutputTexture;
    perImageCB["referenceImage"] =  renderData.pReferenceTexture;

    perImageCB["prevFiltered"] = renderData.fetchInternalTex("LossPrevOutput");
    perImageCB["prevReference"] = renderData.fetchInternalTex("LossPrevReference"); 

    perImageCB["pdaFilteredGaussian"] = mpUtilities->mpdaRawOutputBuffer[0];
    perImageCB["pdaFilteredImage"] = mpUtilities->mpdaRawOutputBuffer[1];

    mLossState.dPass->execute(pRenderContext, mpUtilities->getDummyFullscreenFbo());

    pRenderContext->blit(mpUtilities->getDummyFullscreenFbo()->getColorTexture(0)->getSRV(), renderData.pLossTexture->getRTV());
    pRenderContext->blit(mpUtilities->getDummyFullscreenFbo()->getColorTexture(1)->getSRV(), renderData.pCenterLossTexture->getRTV());
    pRenderContext->blit(mpUtilities->getDummyFullscreenFbo()->getColorTexture(2)->getSRV(), renderData.pGradientLossTexture->getRTV());
    pRenderContext->blit(mpUtilities->getDummyFullscreenFbo()->getColorTexture(3)->getSRV(), renderData.pTemporalLossTexture->getRTV());
    mpUtilities->runCompactingPass(pRenderContext, 0, 9);

    updateLossBuffers(pRenderContext, renderData);

    computeDerivGaussian(pRenderContext);
}

void SVGFPass::computeGaussian(RenderContext* pRenderContext, ref<Texture> tex, ref<Texture> storageLocation, bool saveTextures)
{
    FALCOR_PROFILE(pRenderContext, "Gaussian");

    auto perImageCB = mLossState.sGaussianPass->getRootVar()["PerImageCB"];

    if(saveTextures)
    {
        pRenderContext->blit(tex->getSRV(), mLossState.pGaussianXInput->getRTV());
    }

    perImageCB["image"] = tex;
    perImageCB["yaxis"] = false;
    mLossState.sGaussianPass->execute(pRenderContext, mLossState.pGaussianFbo[0]);

    if(saveTextures)
    {
        pRenderContext->blit(mLossState.pGaussianFbo[0]->getColorTexture(0)->getSRV(), mLossState.pGaussianYInput->getRTV());
    }

    perImageCB["image"] = mLossState.pGaussianFbo[0]->getColorTexture(0);
    perImageCB["yaxis"] = true;

    mLossState.sGaussianPass->execute(pRenderContext, mLossState.pGaussianFbo[1]);

    pRenderContext->blit(mLossState.pGaussianFbo[1]->getColorTexture(0)->getSRV(), storageLocation->getRTV());
}


void SVGFPass::computeDerivGaussian(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "Bwd Gaussian");

    mpUtilities->setPatchingState(mLossState.dGaussianPass);

    auto perImageCB = mLossState.dGaussianPass->getRootVar()["PerImageCB"];
    auto perImageCB_D = mLossState.dGaussianPass->getRootVar()["PerImageCB_D"];

    mpUtilities->clearRawOutputBuffer(pRenderContext, 0);
    perImageCB_D["drIllumination"] = mpUtilities->mpdrCompactedBuffer[0];

    //perImageCB["image"] = mLossState.pGaussianYInput;
    perImageCB["yaxis"] = true;
    perImageCB["pdaIllumination"] = mpUtilities->mpdaRawOutputBuffer[0];
    mLossState.dGaussianPass->execute(pRenderContext, mpUtilities->getDummyFullscreenFbo());

    mpUtilities->runCompactingPass(pRenderContext, 0, 11);

    //perImageCB["image"] = mLossState.pGaussianXInput;
    perImageCB["yaxis"] = false;
    perImageCB["pdaIllumination"] = mpUtilities->mpdaRawOutputBuffer[1];
    mLossState.dGaussianPass->execute(pRenderContext, mpUtilities->getDummyFullscreenFbo());

    // we have the extra derivative from the loss pass
    // not a great way to encapsulate stuff but whatever
    mpUtilities->runCompactingPass(pRenderContext, 1, 12);
}

void SVGFPass::computeDerivFinalModulate(RenderContext* pRenderContext, SVGFRenderData& svgfrd)
{
    FALCOR_PROFILE(pRenderContext, "Bwd Final Modulate");

    mpUtilities->setPatchingState(mFinalModulateState.dPass);

    pRenderContext->clearUAV(mFinalModulateState.pdaIllumination->getUAV().get(), Falcor::uint4(0));

    auto perImageCB = mFinalModulateState.dPass->getRootVar()["PerImageCB"];
    perImageCB["gAlbedo"] = svgfrd.pAlbedoTexture;
    perImageCB["gEmission"] = svgfrd.pEmissionTexture;
    perImageCB["gIllumination"] = svgfrd.fetchInternalTex("FinalModulateFinalFiltered");
    perImageCB["daIllumination"] = mFinalModulateState.pdaIllumination;

    auto perImageCB_D = mFinalModulateState.dPass->getRootVar()["PerImageCB_D"];
    perImageCB_D["drFilteredImage"] = mpUtilities->mpdrCompactedBuffer[1];

    mFinalModulateState.dPass->execute(pRenderContext, mpDerivativeVerifyFbo);

    svgfrd.fetchBufTable("AtrousInIllumination") = mFinalModulateState.pdaIllumination;

    
    auto conversionCB = bufferToTexturePass->getRootVar()["ConversionCB"];
    conversionCB["drIllumination"] = mpUtilities->mpdrCompactedBuffer[1];
    conversionCB["index"] = 0;
    bufferToTexturePass->execute(pRenderContext, bufferToTextureFbo);
}

void SVGFPass::computeAtrousDecomposition(RenderContext* pRenderContext, SVGFRenderData& svgfrd, bool updateInternalBuffers)
{
    mpAtrousSubpass->computeEvaluation(pRenderContext, svgfrd, updateInternalBuffers);
}

void SVGFPass::computeDerivAtrousDecomposition(RenderContext* pRenderContext, SVGFRenderData& svgfrd)
{
    mpAtrousSubpass->computeBackPropagation(pRenderContext, svgfrd);
}

void SVGFPass::computeFilteredMoments(RenderContext* pRenderContext, SVGFRenderData& svgfrd)
{
    FALCOR_PROFILE(pRenderContext, "Filter Moments");

    auto perImageCB = mFilterMomentsState.sPass->getRootVar()["PerImageCB"];

    perImageCB["gIllumination"]     = mpCurReprojFbo->getColorTexture(0);
    perImageCB["gMoments"]          = mpCurReprojFbo->getColorTexture(1);
    perImageCB["gHistoryLength"]    = mpCurReprojFbo->getColorTexture(2);
    perImageCB["gLinearZAndNormal"] = mpLinearZAndNormalFbo->getColorTexture(0);

    perImageCB["dvSigmaL"] = mFilterMomentsState.mSigma.dv.x;
    perImageCB["dvSigmaZ"] = mFilterMomentsState.mSigma.dv.y;
    perImageCB["dvSigmaN"] = mFilterMomentsState.mSigma.dv.z;

    perImageCB["dvLuminanceParams"] = mFilterMomentsState. mLuminanceParams.dv;
    perImageCB["dvVarianceBoostFactor"] = mFilterMomentsState.mVarianceBoostFactor.dv;

    for (int i = 0; i < 3; i++) {
        perImageCB["dvWeightFunctionParams"][i] = mFilterMomentsState.mWeightFunctionParams.dv[i];
    }

    mFilterMomentsState.sPass->execute(pRenderContext, mpPingPongFbo[0]);

    // save our buffers for the derivative apss
    svgfrd.saveInternalTex(pRenderContext, "FilterMomentsIllum", mpCurReprojFbo->getColorTexture(0), false);
    svgfrd.saveInternalTex(pRenderContext, "FilterMomentsMoments", mpCurReprojFbo->getColorTexture(1), false);
    svgfrd.saveInternalTex(pRenderContext, "FilterMomentsHistoryLength", mpCurReprojFbo->getColorTexture(2), false);

    svgfrd.fetchTexTable("AtrousInputIllumination") = mpPingPongFbo[0]->getColorTexture(0);

    pRenderContext->blit(mpPingPongFbo[0]->getColorTexture(0)->getSRV(), mpFilteredPastFbo->getRenderTargetView(4));
    pRenderContext->blit(mpPingPongFbo[0]->getColorTexture(0)->getSRV(), svgfrd.fetchTexTable("FilteredPast4")->getRTV());
}

void SVGFPass::computeDerivFilteredMoments(RenderContext* pRenderContext, SVGFRenderData& svgfrd)
{
    FALCOR_PROFILE(pRenderContext, "Bwd Filter Moments");

    mpUtilities->setPatchingState(mFilterMomentsState.dPass);

    auto perImageCB = mFilterMomentsState.dPass->getRootVar()["PerImageCB"];

    perImageCB["gIllumination"]     = svgfrd.fetchInternalTex("FilterMomentsIllum");
    perImageCB["gMoments"]          = svgfrd.fetchInternalTex("FilterMomentsMoments");
    perImageCB["gHistoryLength"]    = svgfrd.fetchInternalTex("FilterMomentsHistoryLength");
    perImageCB["gLinearZAndNormal"] = svgfrd.fetchInternalTex("LinearZAndNormalTex");

    mpUtilities->clearRawOutputBuffer(pRenderContext, 0);
    mpUtilities->clearRawOutputBuffer(pRenderContext, 1);

    perImageCB["daIllumination"]     = mpUtilities->mpdaRawOutputBuffer[0];
    perImageCB["daMoments"]          = mpUtilities->mpdaRawOutputBuffer[1];
    perImageCB["daHistoryLen"]    = mFilterMomentsState.pdaHistoryLen;

    perImageCB["dvSigmaL"] = mFilterMomentsState.mSigma.dv.x;
    perImageCB["dvSigmaZ"] = mFilterMomentsState.mSigma.dv.y;
    perImageCB["dvSigmaN"] = mFilterMomentsState.mSigma.dv.z;

    perImageCB["dvLuminanceParams"] =mFilterMomentsState.mLuminanceParams.dv;
    perImageCB["dvVarianceBoostFactor"] = mFilterMomentsState.mVarianceBoostFactor.dv;

    for (int i = 0; i < 3; i++) {
        perImageCB["dvWeightFunctionParams"][i] = mFilterMomentsState.mWeightFunctionParams.dv[i];
    }

    auto perImageCB_D = mFilterMomentsState.dPass->getRootVar()["PerImageCB_D"];

    perImageCB_D["drIllumination"] = svgfrd.fetchBufTable("FilterMomentsInIllumination");

    perImageCB_D["daSigma"] = mFilterMomentsState.mSigma.da;
    perImageCB_D["daVarianceBoostFactor"] = mFilterMomentsState.mVarianceBoostFactor.da;
    perImageCB_D["daLuminanceParams"] = mFilterMomentsState.mLuminanceParams.da;
    perImageCB_D["daWeightFunctionParams"] = mFilterMomentsState.mWeightFunctionParams.da;

    mFilterMomentsState.dPass->execute(pRenderContext, mpUtilities->getDummyFullscreenFbo());

    mpUtilities->runCompactingPass(pRenderContext, 0, 50);
    mpUtilities->runCompactingPass(pRenderContext, 1, 49);
}

void SVGFPass::computeReprojection(RenderContext* pRenderContext, SVGFRenderData& svgfrd)
{
    FALCOR_PROFILE(pRenderContext, "Reproj");

    //svgfrd.changeTextureTimeframe(pRenderContext, "ReprojPastFiltered", mpFilteredPastFbo->getColorTexture(0));
    svgfrd.changeTextureTimeframe(pRenderContext, "ReprojPrevIllum", mpPrevReprojFbo->getColorTexture(0));
    svgfrd.changeTextureTimeframe(pRenderContext, "ReprojPrevMoments", mpPrevReprojFbo->getColorTexture(1));
    svgfrd.changeTextureTimeframe(pRenderContext, "ReprojPrevHistoryLength", mpPrevReprojFbo->getColorTexture(2));
    svgfrd.changeTextureTimeframe(pRenderContext, "ReprojPrevTemporalAccum", mpPrevReprojFbo->getColorTexture(3));

    for (int i = 0; i < 5; i++)
    {
        svgfrd.changeTextureTimeframe(pRenderContext, "ReprojPrevFiltered" + std::to_string(i), mpFilteredPastFbo->getColorTexture(i));
    }

    auto perImageCB = mReprojectState.sPass->getRootVar()["PerImageCB"];

    // Setup textures for our reprojection shader pass
    perImageCB["gMotion"] = svgfrd.pMotionVectorTexture;
    perImageCB["gColor"] = svgfrd.pColorTexture;
    perImageCB["gEmission"] = svgfrd.pEmissionTexture;
    perImageCB["gAlbedo"] = svgfrd.pAlbedoTexture;
    perImageCB["gPositionNormalFwidth"] = svgfrd.pPosNormalFwidthTexture;
    perImageCB["gPrevIllum"] = mpPrevReprojFbo->getColorTexture(0);
    perImageCB["gPrevTemporalAccum"] = mpPrevReprojFbo->getColorTexture(3);
    perImageCB["gPrevMoments"] = mpPrevReprojFbo->getColorTexture(1);
    perImageCB["gLinearZAndNormal"] = mpLinearZAndNormalFbo->getColorTexture(0);
    perImageCB["gPrevLinearZAndNormal"] = svgfrd.pPrevLinearZAndNormalTexture;
    perImageCB["gPrevHistoryLength"] = mpPrevReprojFbo->getColorTexture(2);

    for (int i = 0; i < 5; i++)
    {
        perImageCB["gPrevFiltered"][i] = mpFilteredPastFbo->getColorTexture(i);
    }


    // Setup variables for our reprojection pass
    perImageCB["dvAlpha"] = mReprojectState.mAlpha.dv;
    perImageCB["dvMomentsAlpha"] = mReprojectState.mMomentsAlpha.dv;

    perImageCB["dvLuminanceParams"] = mReprojectState.mLuminanceParams.dv;

    for (int i = 0; i < 3; i++) {
        perImageCB["dvReprojKernel"][i] = mReprojectState.mKernel.dv[i];
    }

    for (int i = 0; i < 4; i++) {
        perImageCB["dvReprojParams"][i] = mReprojectState.mParams.dv[i];
    }

    for(int i = 0; i < kNumReprojectionMlpWeights; i++)
    {
        perImageCB["dvMlpWeights"][i] = mReprojectState.mTemporalMlpWeights.dv[i];
    }

    mReprojectState.sPass->execute(pRenderContext, mpCurReprojFbo);

    // save a copy of our past filtration for backwards differentiation
    svgfrd.saveInternalTex(pRenderContext, "ReprojPastFiltered", mpFilteredPastFbo->getColorTexture(0), true);
    svgfrd.saveInternalTex(pRenderContext, "ReprojPrevIllum", mpPrevReprojFbo->getColorTexture(0), true);
    svgfrd.saveInternalTex(pRenderContext, "ReprojPrevMoments", mpPrevReprojFbo->getColorTexture(1), true);
    svgfrd.saveInternalTex(pRenderContext, "ReprojPrevHistoryLength", mpPrevReprojFbo->getColorTexture(2), true);
    svgfrd.saveInternalTex(pRenderContext, "ReprojPrevTemporalAccum", mpPrevReprojFbo->getColorTexture(3), true);

    for (int i = 0; i < 5; i++)
    {
        svgfrd.saveInternalTex(pRenderContext, "ReprojPrevFiltered" + std::to_string(i), mpFilteredPastFbo->getColorTexture(i), true);
        svgfrd.fetchTexTable("FilteredPast" + std::to_string(i)) = mpFilteredPastFbo->getColorTexture(i);
    }

    svgfrd.fetchTexTable("ReprojOutputCurIllum") = mpCurReprojFbo->getColorTexture(0);

    // prevent segfauilt
    svgfrd.fetchBufTable("ReprojOutIllum") = mReprojectState.pdaIllumination;
    svgfrd.fetchBufTable("ReprojOutMoments") = mReprojectState.pdaMoments;
    svgfrd.fetchBufTable("ReprojOutHistoryLength") = mReprojectState.pdaHistoryLength;
}

void SVGFPass::computeDerivReprojection(RenderContext* pRenderContext, SVGFRenderData& svgfrd)
{
    FALCOR_PROFILE(pRenderContext, "Bwd Reproj");

    mpUtilities->setPatchingState(mReprojectState.dPass);

    mpUtilities->clearRawOutputBuffer(pRenderContext, 0);
    mpUtilities->clearRawOutputBuffer(pRenderContext, 1);



    auto perImageCB = mReprojectState.dPass->getRootVar()["PerImageCB"];

    // Setup textures for our reprojection shader pass
    perImageCB["gMotion"]        = svgfrd.pMotionVectorTexture;
    perImageCB["gColor"]         = svgfrd.pColorTexture;
    perImageCB["gEmission"]      = svgfrd.pEmissionTexture;
    perImageCB["gAlbedo"]        = svgfrd.pAlbedoTexture;
    perImageCB["gPositionNormalFwidth"] = svgfrd.pPosNormalFwidthTexture;
    perImageCB["gPrevIllum"]     = svgfrd.fetchInternalTex("ReprojPrevIllum");
    perImageCB["gPrevTemporalAccum"]     = svgfrd.fetchInternalTex("ReprojPrevTemporalAccum");
    perImageCB["gPrevMoments"]   = svgfrd.fetchInternalTex("ReprojPrevMoments");
    perImageCB["gLinearZAndNormal"]       = svgfrd.fetchInternalTex("LinearZAndNormalTex");
    perImageCB["gPrevLinearZAndNormal"]   = svgfrd.fetchInternalTex("LinearZAndNormalPrevTex");
    perImageCB["gPrevHistoryLength"] = svgfrd.fetchInternalTex("ReprojPrevHistoryLength");

    for (int i = 0; i < 5; i++)
    {
        perImageCB["gPrevFiltered"][i] = svgfrd.fetchInternalTex("ReprojPrevFiltered" + std::to_string(i));
    }

    // Setup variables for our reprojection pass
    perImageCB["dvAlpha"] = mReprojectState.mAlpha.dv;
    perImageCB["dvMomentsAlpha"] = mReprojectState.mMomentsAlpha.dv;

    perImageCB["dvLuminanceParams"] = mReprojectState.mLuminanceParams.dv;

    for (int i = 0; i < 3; i++) {
        perImageCB["dvReprojKernel"][i] = mReprojectState.mKernel.dv[i];
    }

    for (int i = 0; i < 4; i++) {
        perImageCB["dvReprojParams"][i] = mReprojectState.mParams.dv[i];
    }

    for(int i = 0; i < kNumReprojectionMlpWeights; i++)
    {
        perImageCB["dvMlpWeights"][i] = mReprojectState.mTemporalMlpWeights.dv[i];
    }
    perImageCB["daMlpWeights"] =  mReprojectState.mTemporalMlpWeights.da;

    mpUtilities->combineBuffers(pRenderContext, 0, mpUtilities->mpdrCompactedBuffer[1], mReprojectState.pdaMoments);
    mpUtilities->combineBuffers(pRenderContext, 1, mFilterMomentsState.pdaHistoryLen, mReprojectState.pdaHistoryLength);
    pRenderContext->clearUAV(mReprojectState.pdaMoments->getUAV().get(), float4(0.0f));
    pRenderContext->clearUAV(mReprojectState.pdaHistoryLength->getUAV().get(), float4(0.0f));

    auto perImageCB_D = mReprojectState.dPass->getRootVar()["PerImageCB_D"];

    perImageCB_D["drIllumination"] = mpUtilities->mpdrCompactedBuffer[0];
    perImageCB_D["drMoments"] = mpUtilities->mpdrCombinedBuffer[0];
    perImageCB_D["drHistoryLen"] = mpUtilities->mpdrCombinedBuffer[1];

    perImageCB_D["daLuminanceParams"] = mReprojectState.mLuminanceParams.da;
    perImageCB_D["daReprojKernel"] = mReprojectState.mKernel.da;
    perImageCB_D["daReprojParams"] = mReprojectState.mParams.da;
    perImageCB_D["daAlpha"] = mReprojectState.mAlpha.da;
    perImageCB_D["daMomentsAlpha"] = mReprojectState.mMomentsAlpha.da;

    perImageCB["daIllumination"] = mpUtilities->mpdaRawOutputBuffer[0];
    perImageCB["daMoments"] = mpUtilities->mpdaRawOutputBuffer[1];
    perImageCB["daHistoryLength"] = mReprojectState.pdaHistoryLength;

    mReprojectState.dPass->execute(pRenderContext, mpUtilities->getDummyFullscreenFbo());

    // now, save the illum, moments, and history length buffers somewhere
    mpUtilities->runCompactingPass(pRenderContext, 0, 9 + 4);
    mpUtilities->runCompactingPass(pRenderContext, 1, 9 + 4);

    pRenderContext->copyBufferRegion(mReprojectState.pdaIllumination.get(), 0, mpUtilities->mpdrCompactedBuffer[0].get(), 0, mpUtilities->mpdrCompactedBuffer[0]->getSize());
    pRenderContext->copyBufferRegion(mReprojectState.pdaMoments.get(), 0, mpUtilities->mpdrCompactedBuffer[1].get(), 0, mpUtilities->mpdrCompactedBuffer[1]->getSize());

    svgfrd.fetchBufTable("ReprojOutIllum") = mReprojectState.pdaIllumination;
    svgfrd.fetchBufTable("ReprojOutMoments") = mReprojectState.pdaMoments;
    svgfrd.fetchBufTable("ReprojOutHistoryLength") = mReprojectState.pdaHistoryLength;
}



#define USE_BUILTIN_PARALLEL_REDUCTION
int SVGFPass::reduceParameter(RenderContext* pRenderContext, ParameterMetaInfo& param, int offset)
{
#ifdef USE_BUILTIN_PARALLEL_REDUCTION
    mpUtilities->setPatchingState(bufferToTexturePass);

    auto conversionCB = bufferToTexturePass->getRootVar()["ConversionCB"];

    int numPasses = (param.mNumElements + 3) / 4;

    conversionCB["drIllumination"] = param.mAccum;
    for(int i = 0; i < numPasses; i++)
    {
        conversionCB["index"] = i;
        bufferToTexturePass->execute(pRenderContext, bufferToTextureFbo);
        mpParallelReduction->execute<float4>(pRenderContext, bufferToTextureFbo->getColorTexture(0), ParallelReduction::Type::Sum, nullptr, mReadbackBuffer[0], offset + i * sizeof(float4));
    }

    return offset + numPasses * sizeof(float4);
#else
    const int K_NUM_ITERATIONS = 3;

    int numRemaining = numPixels;
    for (int i = 0; i < K_NUM_ITERATIONS; i++)
    {
        // clear the output buffer
        pRenderContext->clearUAV(pdaPingPongSumBuffer[1]->getUAV().get(), Falcor::uint4(0));

        auto summingCB = summingPass->getRootVar()["SummingCB"];

        summingCB["srcBuf"] = (i == 0 ? param.da : pdaPingPongSumBuffer[0]);
        summingCB["srcOffset"] = 0;

        if (i == K_NUM_ITERATIONS - 1)
        {
            // if it is the last pass, write out output to a particular location
            summingCB["dstBuf"] = pdaGradientBuffer;
            summingCB["dstOffset"] = offset;
        }
        else
        {
            summingCB["dstBuf"] = pdaPingPongSumBuffer[1];
            summingCB["dstOffset"] = 0;
        }

        summingPass->execute(pRenderContext, numRemaining, 1);
        // round up divide
        numRemaining = (numRemaining + 127) / 128;

        std::swap(pdaPingPongSumBuffer[0], pdaPingPongSumBuffer[1]);
    }
#endif
}

// Extracts linear z and its derivative from the linear Z texture and packs
// the normal from the world normal texture and packes them into the FBO.
// (It's slightly wasteful to copy linear z here, but having this all
// together in a single buffer is a small simplification, since we make a
// copy of it to refer to in the next frame.)
void SVGFPass::computeLinearZAndNormal(RenderContext* pRenderContext, SVGFRenderData& svgfrd)
{
    FALCOR_PROFILE(pRenderContext, "Linear Z and Normal");


    auto perImageCB = mPackLinearZAndNormalState.sPass->getRootVar()["PerImageCB"];
    perImageCB["gLinearZ"] = svgfrd.pLinearZTexture;
    perImageCB["gNormal"] = svgfrd.pWorldNormalTexture;

    mPackLinearZAndNormalState.sPass->execute(pRenderContext, mpLinearZAndNormalFbo);

    svgfrd.saveInternalTex(pRenderContext, "LinearZAndNormalTex", mpLinearZAndNormalFbo->getColorTexture(0), false);
    svgfrd.fetchTexTable("gLinearZAndNormal") = svgfrd.fetchInternalTex("LinearZAndNormalTex"); // point to the unchanging one

    svgfrd.saveInternalTex(pRenderContext, "LinearZAndNormalPrevTex", svgfrd.pPrevLinearZAndNormalTexture, true);
}

void SVGFPass::renderUI(Gui::Widgets& widget)
{
    float dummyVal = 0.0f;

    int dirty = 0;
    dirty |= (int)widget.checkbox("Enable SVGF", mFilterEnabled);

    widget.text("");
    widget.text("Number of filter iterations.  Which");
    widget.text("    iteration feeds into future frames?");
    dirty |= (int)widget.var("Iterations", mFilterIterations, 1, 10, 1);
    dirty |= (int)widget.var("Feedback", mFeedbackTap, -1, mFilterIterations - 2, 1);

    widget.var("mDI", mDerivativeIteration, 0, mFilterIterations - 1, 1);

    widget.text("");
    widget.text("Contol edge stopping on bilateral fitler");
    dirty |= (int)widget.var("For Color", dummyVal, 0.0f, 10000.0f, 0.01f);  // pass in sigma l as dummy var
    dirty |= (int)widget.var("For Normal", dummyVal, 0.001f, 1000.0f, 0.2f);

    widget.text("");
    widget.text("How much history should be used?");
    widget.text("    (alpha; 0 = full reuse; 1 = no reuse)");
    dirty |= (int)widget.var("Alpha", mReprojectState.mAlpha.dv, 0.0f, 1.0f, 0.001f);
    dirty |= (int)widget.var("Moments Alpha", mReprojectState.mMomentsAlpha.dv, 0.0f, 1.0f, 0.001f);

    widget.checkbox("Keep running test", mKeepRunningKpcnnTest);

    if (dirty)
        mBuffersNeedClear = true;
}

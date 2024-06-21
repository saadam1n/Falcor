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

/*
TODO:
- clean up shaders
- clean up UI: tooltips, etc.
- handle skybox pixels
- enum for fbo channel indices
*/

namespace
{
    // Shader source files
    const char kPackLinearZAndNormalShader[] = "RenderPasses/SVGFPass/SVGFPackLinearZAndNormal.ps.slang";

    const char kReprojectShaderS[]            = "RenderPasses/SVGFPass/SVGFReprojectS.ps.slang";
    const char kReprojectShaderD[]            = "RenderPasses/SVGFPass/SVGFReprojectD.ps.slang";

    const char kAtrousShaderS[]               = "RenderPasses/SVGFPass/SVGFAtrousS.ps.slang";
    const char kAtrousShaderD[]               = "RenderPasses/SVGFPass/SVGFAtrousD.ps.slang";
    const char kAtrousShaderF[]               = "RenderPasses/SVGFPass/SVGFAtrousF.ps.slang";

    const char kFilterMomentShaderS[]         = "RenderPasses/SVGFPass/SVGFFilterMomentsS.ps.slang";
    const char kFilterMomentShaderD[]         = "RenderPasses/SVGFPass/SVGFFilterMomentsD.ps.slang";

    const char kFinalModulateShaderS[]        = "RenderPasses/SVGFPass/SVGFFinalModulateS.ps.slang";
    const char kFinalModulateShaderD[]        = "RenderPasses/SVGFPass/SVGFFinalModulateD.ps.slang";

    const char kDerivativeVerifyShader[]        = "RenderPasses/SVGFPass/SVGFDerivativeVerify.ps.slang";

    // Names of valid entries in the parameter dictionary.
    const char kEnabled[] = "Enabled";
    const char kIterations[] = "Iterations";
    const char kFeedbackTap[] = "FeedbackTap";
    const char kVarianceEpsilon[] = "VarianceEpsilon";
    const char kPhiColor[] = "PhiColor";
    const char kPhiNormal[] = "PhiNormal";
    const char kAlpha[] = "Alpha";
    const char kMomentsAlpha[] = "MomentsAlpha";

// Input buffer names
const char kInputBufferAlbedo[] = "Albedo";
const char kInputBufferColor[] = "Color";
const char kInputBufferEmission[] = "Emission";
const char kInputBufferWorldPosition[] = "WorldPosition";
const char kInputBufferWorldNormal[] = "WorldNormal";
const char kInputBufferPosNormalFwidth[] = "PositionNormalFwidth";
const char kInputBufferLinearZ[] = "LinearZ";
const char kInputBufferMotionVector[] = "MotionVec";

// Internal buffer names
const char kInternalBufferPreviousLinearZAndNormal[] = "Previous Linear Z and Packed Normal";
const char kInternalBufferPreviousLighting[] = "Previous Lighting";
const char kInternalBufferPreviousMoments[] = "Previous Moments";

    // Output buffer name
    const char kOutputBufferFilteredImage[] = "Filtered image";
    const char kOutputDebugBuffer[] = "DebugBuf";
    const char kOutputDerivVerifyBuf[] = "DerivVerify";
    const char kOutputFuncLower[] = "FuncLower";
    const char kOutputFuncUpper[] = "FuncUpper";
    const char kOutputFdCol[] = "FdCol";
    const char kOutputBdCol[] = "BdCol";

        // set common stuff first
    const size_t screenWidth = 1920;
    const size_t screenHeight = 1080;
    const size_t numPixels = screenWidth * screenHeight;

    }

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, SVGFPass>();
}

SVGFPass::SVGFPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
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

    mFilterIterations = 2;
    mFeedbackTap = -1;
    mDerivativeInteration = 0;

    mpPackLinearZAndNormal = FullScreenPass::create(mpDevice, kPackLinearZAndNormalShader);
    mpReprojection = FullScreenPass::create(mpDevice, kReprojectShaderS);
    mpAtrous = FullScreenPass::create(mpDevice, kAtrousShaderS);
    mpFilterMoments = FullScreenPass::create(mpDevice, kFilterMomentShaderS);
    mpFinalModulate = FullScreenPass::create(mpDevice, kFinalModulateShaderS);

    mpDerivativeVerify = FullScreenPass::create(mpDevice, kDerivativeVerifyShader);
    mpFuncOutputLower =  make_ref<Texture>(pDevice, Resource::Type::Texture2D, ResourceFormat::RGBA32Float, screenWidth, screenHeight,  1, 1, 1, 1, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource, nullptr);
    mpFuncOutputUpper =  make_ref<Texture>(pDevice, Resource::Type::Texture2D, ResourceFormat::RGBA32Float, screenWidth, screenHeight,  1, 1, 1, 1, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource, nullptr);

    mFinalModulateState.dPass = FullScreenPass::create(mpDevice, kFinalModulateShaderD);
    mAtrousState.dPass = FullScreenPass::create(mpDevice, kAtrousShaderD);
    mFilterMomentsState.dPass = FullScreenPass::create(mpDevice, kFilterMomentShaderD);
    mReprojectState.dPass = FullScreenPass::create(mpDevice, kReprojectShaderD);

    FALCOR_ASSERT(
        mFinalModulateState.dPass &&
        mAtrousState.dPass &&
        mFilterMomentsState.dPass &&
        mReprojectState.dPass
    );

    FALCOR_ASSERT(mpPackLinearZAndNormal && mpReprojection && mpAtrous && mpFilterMoments && mpFinalModulate);

    float3 dvLuminanceParams = float3(0.3333);

    float   dvSigmaL              = 1.0f;
    float   dvSigmaZ              = 1.0;
    float   dvSigmaN              = 128.0f;
    float   dvAlpha               = 0.05f;
    float   dvMomentsAlpha        = 0.2f;

    float dvWeightFunctionParams[3] {1.0, 1.0, 1.0};

    // set pack linear z and normal params


    // set reproj params
    mReprojectState.dvLuminanceParams = dvLuminanceParams;
    mReprojectState.dvAlpha = dvAlpha;
    mReprojectState.dvMomentsAlpha = dvMomentsAlpha;

    mReprojectState.dvParams[0] = 32.0;
    mReprojectState.dvParams[1] = 1.0;
    mReprojectState.dvParams[2] = 10.0;
    mReprojectState.dvParams[3] = 16.0;

    mReprojectState.dvKernel[0] = 1.0;
    mReprojectState.dvKernel[1] = 1.0;
    mReprojectState.dvKernel[2] = 1.0;

    mReprojectState.pdaLuminanceParams = createAccumulationBuffer(pDevice);
    mReprojectState.pdaReprojKernel = createAccumulationBuffer(pDevice);
    mReprojectState.pdaReprojParams = createAccumulationBuffer(pDevice);
    mReprojectState.pdaAlpha = createAccumulationBuffer(pDevice);
    mReprojectState.pdaMomentsAlpha = createAccumulationBuffer(pDevice);

    mReprojectState.pPrevIllum = createFullscreenTexture(pDevice);

    // set filter moments params
    mFilterMomentsState.dvSigmaL = dvSigmaL;
    mFilterMomentsState.dvSigmaZ = dvSigmaZ;
    mFilterMomentsState.dvSigmaN = dvSigmaN;

    mFilterMomentsState.dvLuminanceParams = dvLuminanceParams;

    for (int i = 0; i < 3; i++) {
        mFilterMomentsState.dvWeightFunctionParams[i] = dvWeightFunctionParams[i];
    }

    mFilterMomentsState.dvVarianceBoostFactor = 4.0;
    mFilterMomentsState.pdaIllumination = createAccumulationBuffer(pDevice);
    mFilterMomentsState.pdaMoments = createAccumulationBuffer(pDevice);

    mFilterMomentsState.pdaSigmaL = createAccumulationBuffer(pDevice);
    mFilterMomentsState.pdaSigmaZ = createAccumulationBuffer(pDevice);
    mFilterMomentsState.pdaSigmaN = createAccumulationBuffer(pDevice);

    mFilterMomentsState.pdaLuminanceParams = createAccumulationBuffer(pDevice);
    mFilterMomentsState.pdaVarianceBoostFactor = createAccumulationBuffer(pDevice);
    mFilterMomentsState.pdaWeightFunctionParams = createAccumulationBuffer(pDevice);

    // Set atrous state vars

    mAtrousState.mIterationState.resize(mFilterIterations);
    for (auto& iterationState : mAtrousState.mIterationState)
    {
        iterationState.dvSigmaL = dvSigmaL;
        iterationState.dvSigmaZ = dvSigmaZ;
        iterationState.dvSigmaN = dvSigmaN;

        for (int i = 0; i < 3; i++) {
            iterationState.dvWeightFunctionParams[i] = dvWeightFunctionParams[i];
        }

        iterationState.dvLuminanceParams = dvLuminanceParams;

        iterationState.dvKernel[0] = 1.0;
        iterationState.dvKernel[1] = 2.0f / 3.0f;
        iterationState.dvKernel[2] = 1.0f / 6.0f;

        iterationState.dvVarianceKernel[0][0] = 1.0 / 4.0;
        iterationState.dvVarianceKernel[0][1] = 1.0 / 8.0;
        iterationState.dvVarianceKernel[1][0] = 1.0 / 8.0;
        iterationState.dvVarianceKernel[1][1] = 1.0 / 16.0;

        iterationState.pgIllumination = createFullscreenTexture(pDevice);


        iterationState.pdaSigma = createAccumulationBuffer(pDevice);
        iterationState.pdaKernel = createAccumulationBuffer(pDevice);
        iterationState.pdaVarianceKernel = createAccumulationBuffer(pDevice);
        iterationState.pdaLuminanceParams = createAccumulationBuffer(pDevice);
        iterationState.pdaWeightFunctionParams = createAccumulationBuffer(pDevice);

        // flaot4/4 color channel-specific derivaitves/9 + 25 derivatives per SVGF iteration
        iterationState.pdaIllumination = createAccumulationBuffer(pDevice, sizeof(float4) * (9 + 25));
    }

    mAtrousState.pdaHistoryLen = createAccumulationBuffer(pDevice);

    // set final modulate state vars
    mFinalModulateState.pdaIllumination = createAccumulationBuffer(pDevice);
    mFinalModulateState.pdrFilteredImage = createAccumulationBuffer(pDevice);


    FALCOR_ASSERT(mFinalModulateState.pdaIllumination &&  mFinalModulateState.pdrFilteredImage);

    mReadbackBuffer = make_ref<Buffer>(pDevice, sizeof(int4) * numPixels, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::ReadBack, nullptr);

    mAtrousState.mSaveIllum = createFullscreenTexture(pDevice, ResourceFormat::RGBA32Int);
}

void SVGFPass::clearBuffers(RenderContext* pRenderContext, const RenderData& renderData)
{
    pRenderContext->clearFbo(mpPingPongFbo[0].get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpPingPongFbo[1].get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpLinearZAndNormalFbo.get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpFilteredPastFbo.get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpCurReprojFbo.get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpPrevReprojFbo.get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpFilteredIlluminationFbo.get(), float4(0), 1.0f, 0, FboAttachmentType::All);

    pRenderContext->clearTexture(renderData.getTexture(kInternalBufferPreviousLinearZAndNormal).get());
    pRenderContext->clearTexture(renderData.getTexture(kInternalBufferPreviousLighting).get());
    pRenderContext->clearTexture(renderData.getTexture(kInternalBufferPreviousMoments).get());

    pRenderContext->clearFbo(mpDerivativeVerifyFbo.get(), float4(0), 1.0f, 0, FboAttachmentType::All);
}

void SVGFPass::allocateFbos(uint2 dim, RenderContext* pRenderContext)
{
    {
        // Screen-size FBOs with 3 MRTs: one that is RGBA32F, one that is
        // RG32F for the luminance moments, and one that is R16F.
        Fbo::Desc desc;
        desc.setSampleCount(0);
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float); // illumination
        desc.setColorTarget(1, Falcor::ResourceFormat::RGBA32Float);   // moments
        desc.setColorTarget(2, Falcor::ResourceFormat::RGBA32Float);    // history length
        desc.setColorTarget(3, Falcor::ResourceFormat::RGBA32Float);    // debug buf
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
        desc.setColorTarget(1, Falcor::ResourceFormat::RGBA32Float);    // debug buf

        mpPingPongFbo[0]  = Fbo::create2D(mpDevice, dim.x, dim.y, desc);
        mpPingPongFbo[1]  = Fbo::create2D(mpDevice, dim.x, dim.y, desc);
        mpFilteredPastFbo = Fbo::create2D(mpDevice, dim.x, dim.y, desc);
        mpFilteredIlluminationFbo = Fbo::create2D(mpDevice, dim.x, dim.y, desc);
        mpFinalFbo = Fbo::create2D(mpDevice, dim.x, dim.y, desc);
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
        // contains a debug buffer for whatever we want to store
        Fbo::Desc desc;
        desc.setSampleCount(0);
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float);
        desc.setColorTarget(1, Falcor::ResourceFormat::RGBA32Int);
        mpDummyFullscreenFbo = Fbo::create2D(mpDevice, dim.x, dim.y, desc);
    }

    mBuffersNeedClear = true;
}


ref<Buffer> SVGFPass::createAccumulationBuffer(ref<Device> pDevice, int bytes_per_elem) {
    return make_ref<Buffer>(pDevice, bytes_per_elem * numPixels, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr);
}

ref<Texture> SVGFPass::createFullscreenTexture(ref<Device> pDevice, ResourceFormat fmt)
{
    return make_ref<Texture>(pDevice, Resource::Type::Texture2D, fmt, screenWidth, screenHeight,  1, 1, 1, 1, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, nullptr);
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

    reflector.addOutput(kOutputBufferFilteredImage, "Filtered image").format(ResourceFormat::RGBA16Float);
    reflector.addOutput(kOutputDebugBuffer, "DebugBuf").format(ResourceFormat::RGBA32Float);
    reflector.addOutput(kOutputDerivVerifyBuf, "Deriv Verify").format(ResourceFormat::RGBA32Float);
    reflector.addOutput(kOutputFuncLower, "Func lower").format(ResourceFormat::RGBA32Float);
    reflector.addOutput(kOutputFuncUpper, "Func upper").format(ResourceFormat::RGBA32Float);
    reflector.addOutput(kOutputFdCol, "FdCol").format(ResourceFormat::RGBA32Float);
    reflector.addOutput(kOutputBdCol, "BdCol").format(ResourceFormat::RGBA32Float);

    return reflector;
}

void SVGFPass::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    allocateFbos(compileData.defaultTexDims, pRenderContext);
    mBuffersNeedClear = true;
}

void SVGFPass::executeWithDerivatives(RenderContext* pRenderContext, const RenderData& renderData, bool shouldCalcDerivatives)
{
    ref<Texture> pAlbedoTexture = renderData.getTexture(kInputBufferAlbedo);
    ref<Texture> pColorTexture = renderData.getTexture(kInputBufferColor);
    ref<Texture> pEmissionTexture = renderData.getTexture(kInputBufferEmission);
    ref<Texture> pWorldPositionTexture = renderData.getTexture(kInputBufferWorldPosition);
    ref<Texture> pWorldNormalTexture = renderData.getTexture(kInputBufferWorldNormal);
    ref<Texture> pPosNormalFwidthTexture = renderData.getTexture(kInputBufferPosNormalFwidth);
    ref<Texture> pLinearZTexture = renderData.getTexture(kInputBufferLinearZ);
    ref<Texture> pMotionVectorTexture = renderData.getTexture(kInputBufferMotionVector);

    ref<Texture> pOutputTexture = renderData.getTexture(kOutputBufferFilteredImage);
    ref<Texture> pDebugTexture = renderData.getTexture(kOutputDebugBuffer);
    ref<Texture> pDerivVerifyTexture = renderData.getTexture(kOutputDerivVerifyBuf);

    FALCOR_ASSERT(
        mpFilteredIlluminationFbo && mpFilteredIlluminationFbo->getWidth() == pAlbedoTexture->getWidth() &&
        mpFilteredIlluminationFbo->getHeight() == pAlbedoTexture->getHeight()
    );

    if (mBuffersNeedClear)
    {
        clearBuffers(pRenderContext, renderData);
        mBuffersNeedClear = false;
    }

    if (mFilterEnabled)
    {
        // Grab linear z and its derivative and also pack the normal into
        // the last two channels of the mpLinearZAndNormalFbo.
        computeLinearZAndNormal(pRenderContext, pLinearZTexture, pWorldNormalTexture);

        // Demodulate input color & albedo to get illumination and lerp in
        // reprojected filtered illumination from the previous frame.
        // Stores the result as well as initial moments and an updated
        // per-pixel history length in mpCurReprojFbo.
        ref<Texture> pPrevLinearZAndNormalTexture =
            renderData.getTexture(kInternalBufferPreviousLinearZAndNormal);
        computeReprojection(pRenderContext, pAlbedoTexture, pColorTexture, pEmissionTexture,
                            pMotionVectorTexture, pPosNormalFwidthTexture,
                            pPrevLinearZAndNormalTexture, pDebugTexture);

        // Do a first cross-bilateral filtering of the illumination and
        // estimate its variance, storing the result into a float4 in
        // mpPingPongFbo[0].  Takes mpCurReprojFbo as input.
        computeFilteredMoments(pRenderContext);

        //pRenderContext->blit(mpPingPongFbo[0]->getColorTexture(1)->getSRV(), pDebugTexture->getRTV());

        // Filter illumination from mpCurReprojFbo[0], storing the result
        // in mpPingPongFbo[0].  Along the way (or at the end, depending on
        // the value of mFeedbackTap), save the filtered illumination for
        // next time into mpFilteredPastFbo.
        computeAtrousDecomposition(pRenderContext, pAlbedoTexture, shouldCalcDerivatives);

        // Compute albedo * filtered illumination and add emission back in.
        auto perImageCB = mpFinalModulate->getRootVar()["PerImageCB"];
        perImageCB["gAlbedo"] = pAlbedoTexture;
        perImageCB["gEmission"] = pEmissionTexture;
        perImageCB["gIllumination"] = mpPingPongFbo[0]->getColorTexture(0);
        mpFinalModulate->execute(pRenderContext, mpFinalFbo);

        if (shouldCalcDerivatives)
        {
            computeDerivatives(pRenderContext, renderData);

            computeDerivVerification(pRenderContext);
            pRenderContext->blit(mpDerivativeVerifyFbo->getColorTexture(0)->getSRV(), pDerivVerifyTexture->getRTV());

            // Swap resources so we're ready for next frame.
            // only do it though if we are calculating derivaitves so we don't screw up our results from the finite diff pass
            std::swap(mpCurReprojFbo, mpPrevReprojFbo);
            pRenderContext->blit(mpLinearZAndNormalFbo->getColorTexture(0)->getSRV(),
                                    pPrevLinearZAndNormalTexture->getRTV());

            // Blit into the output texture.
            pRenderContext->blit(mpFinalFbo->getColorTexture(0)->getSRV(), pOutputTexture->getRTV());
        }

    }
    else
    {
        pRenderContext->blit(pColorTexture->getSRV(), pOutputTexture->getRTV());
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
    if(!pScene)
        return;

    mDelta = 0.05f;

    float& valToChange = mAtrousState.mIterationState[mDerivativeInteration].dvKernel[0];
    float oldval = valToChange;

    valToChange = oldval - mDelta;
    executeWithDerivatives(pRenderContext, renderData, false);
    pRenderContext->blit(mpFinalFbo->getColorTexture(0)->getSRV(), mpFuncOutputLower->getRTV());
    pRenderContext->blit(mpFinalFbo->getColorTexture(0)->getSRV(), renderData.getTexture(kOutputFuncLower)->getRTV());


    valToChange = oldval + mDelta;
    executeWithDerivatives(pRenderContext, renderData, false);
    pRenderContext->blit(mpFinalFbo->getColorTexture(0)->getSRV(), mpFuncOutputUpper->getRTV());
    pRenderContext->blit(mpFinalFbo->getColorTexture(0)->getSRV(),  renderData.getTexture(kOutputFuncUpper)->getRTV());

    valToChange = oldval;
    executeWithDerivatives(pRenderContext, renderData, true);
    pRenderContext->blit(mpDerivativeVerifyFbo->getColorTexture(1)->getSRV(),  renderData.getTexture(kOutputFdCol)->getRTV());
    pRenderContext->blit(mpDerivativeVerifyFbo->getColorTexture(2)->getSRV(),  renderData.getTexture(kOutputBdCol)->getRTV());

    pRenderContext->blit(mAtrousState.mSaveIllum->getSRV(),   renderData.getTexture(kOutputDebugBuffer)->getRTV());

    std::cout << "Fwd Diff Sum:\t" << getTexSum(pRenderContext, mpDerivativeVerifyFbo->getColorTexture(1)) << std::endl;
    std::cout << "Bwd Diff Sum:\t" << getTexSum(pRenderContext, mpDerivativeVerifyFbo->getColorTexture(2)) << std::endl;
    std::cout << std::endl;
}

void SVGFPass::computeDerivVerification(RenderContext* pRenderContext)
{
    auto perImageCB = mpDerivativeVerify->getRootVar()["PerImageCB"];

    perImageCB["drBackwardsDiffBuffer"] = mAtrousState.mIterationState[mDerivativeInteration].pdaKernel;
    perImageCB["gFuncOutputLower"] = mpFuncOutputLower;
    perImageCB["gFuncOutputUpper"] = mpFuncOutputUpper;
    perImageCB["delta"] = mDelta;

    mpDerivativeVerify->execute(pRenderContext, mpDerivativeVerifyFbo);
}


// I'll move parts of this off to other function as need be
void SVGFPass::computeDerivatives(RenderContext* pRenderContext, const RenderData& renderData)
{
    ref<Texture> pOutputTexture = renderData.getTexture(kOutputBufferFilteredImage);
    ref<Texture> pAlbedoTexture = renderData.getTexture(kInputBufferAlbedo);
    ref<Texture> pEmissionTexture = renderData.getTexture(kInputBufferEmission);
    ref<Texture> pIllumTexture = mpPingPongFbo[0]->getColorTexture(0);
    ref<Texture> pColorTexture = renderData.getTexture(kInputBufferColor);
    ref<Texture> pPosNormalFwidthTexture = renderData.getTexture(kInputBufferPosNormalFwidth);
    ref<Texture> pLinearZTexture = renderData.getTexture(kInputBufferLinearZ);
    ref<Texture> pMotionVectorTexture = renderData.getTexture(kInputBufferMotionVector);
    ref<Texture> pDebugTexture = renderData.getTexture(kOutputDebugBuffer);

    if (mFilterEnabled) {
        computeDerivFinalModulate(pRenderContext, pOutputTexture, pIllumTexture, pAlbedoTexture, pEmissionTexture);

        // now, the derivative is stored in mFinalModulateState.pdaIllum
        // now, we will have to computer the atrous decomposition reverse
        // the atrous decomp has multiple stages
        // each stage outputs the exact same result - a color
        // we need to use that color and its derivative to feed the previous pass
        // ideally, we will want a buffer and specific variables for each stage
        // right now, I'll just set up the buffers

        computeDerivAtrousDecomposition(pRenderContext, pAlbedoTexture, pOutputTexture);

        computeDerivFilteredMoments(pRenderContext);

        computeDerivReprojection(pRenderContext, pAlbedoTexture, pColorTexture, pEmissionTexture, pMotionVectorTexture, pPosNormalFwidthTexture, pLinearZTexture, pDebugTexture);
    }
}

void SVGFPass::computeDerivFinalModulate(RenderContext* pRenderContext, ref<Texture> pOutputTexture, ref<Texture> pIllumination, ref<Texture> pAlbedoTexture, ref<Texture> pEmissionTexture) {
    pRenderContext->clearUAV(mFinalModulateState.pdaIllumination->getUAV().get(), Falcor::uint4(0));

    auto perImageCB = mFinalModulateState.dPass->getRootVar()["PerImageCB"];
    perImageCB["gAlbedo"] = pAlbedoTexture;
    perImageCB["gEmission"] = pEmissionTexture;
    perImageCB["gIllumination"] = mpPingPongFbo[0]->getColorTexture(0);
    perImageCB["daIllumination"] = mFinalModulateState.pdaIllumination;

    auto perImageCB_D = mFinalModulateState.dPass->getRootVar()["PerImageCB_D"];
    perImageCB_D["drFilteredImage"] =  mFinalModulateState.pdrFilteredImage; // uh placehold for now

    mFinalModulateState.dPass->execute(pRenderContext, mpDerivativeVerifyFbo);
}

void SVGFPass::computeAtrousDecomposition(RenderContext* pRenderContext, ref<Texture> pAlbedoTexture, bool nonFiniteDiffPass)
{
    auto perImageCB = mpAtrous->getRootVar()["PerImageCB"];

    perImageCB["gAlbedo"] = pAlbedoTexture;
    perImageCB["gHistoryLength"] = mpCurReprojFbo->getColorTexture(2);
    perImageCB["gLinearZAndNormal"] = mpLinearZAndNormalFbo->getColorTexture(0);

    for (int iteration = 0; iteration < mFilterIterations; iteration++)
    {

        auto& curIterationState = mAtrousState.mIterationState[iteration];

        perImageCB["dvSigmaL"] = curIterationState.dvSigmaL;
        perImageCB["dvSigmaZ"] = curIterationState.dvSigmaZ;
        perImageCB["dvSigmaN"] = curIterationState.dvSigmaN;

        perImageCB["dvLuminanceParams"] = curIterationState.dvLuminanceParams;

        for (int i = 0; i < 3; i++) {
            perImageCB["dvWeightFunctionParams"][i] = curIterationState.dvWeightFunctionParams[i];
        }

        for (int i = 0; i < 3; i++) {
            perImageCB["dvKernel"][i] = curIterationState.dvKernel[i];
        }

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                perImageCB["dvVarianceKernel"][i][j] = curIterationState.dvVarianceKernel[i][j];
            }
        }


        ref<Fbo> curTargetFbo = mpPingPongFbo[1];
        // keep a copy of our input for backwards differation
        if(nonFiniteDiffPass)
            pRenderContext->blit(mpPingPongFbo[0]->getColorTexture(0)->getSRV(), curIterationState.pgIllumination->getRTV());

        perImageCB["gIllumination"] = mpPingPongFbo[0]->getColorTexture(0);
        perImageCB["gStepSize"] = 1 << iteration;


        mpAtrous->execute(pRenderContext, curTargetFbo);

        // store the filtered color for the feedback path
        if (nonFiniteDiffPass && iteration == std::min(mFeedbackTap, mFilterIterations - 1))
        {
            pRenderContext->blit(curTargetFbo->getColorTexture(0)->getSRV(), mpFilteredPastFbo->getRenderTargetView(0));
        }

        std::swap(mpPingPongFbo[0], mpPingPongFbo[1]);
    }

    if (nonFiniteDiffPass && mFeedbackTap < 0)
    {
        pRenderContext->blit(mpCurReprojFbo->getColorTexture(0)->getSRV(), mpFilteredPastFbo->getRenderTargetView(0));
    }
}

void SVGFPass::computeDerivAtrousDecomposition(RenderContext* pRenderContext, ref<Texture> pAlbedoTexture, ref<Texture> pOutputTexture)
{
    auto perImageCB = mAtrousState.dPass->getRootVar()["PerImageCB"];
    auto perImageCB_D = mAtrousState.dPass->getRootVar()["PerImageCB_D"];

    pRenderContext->clearUAV(mAtrousState.pdaHistoryLen->getUAV().get(), Falcor::uint4(0));

    perImageCB["gAlbedo"]        = pAlbedoTexture;
    perImageCB["gHistoryLength"] = mpCurReprojFbo->getColorTexture(2);
    perImageCB["gLinearZAndNormal"]       = mpLinearZAndNormalFbo->getColorTexture(0);


    for (int iteration = mFilterIterations - 1; iteration >= 0; iteration--)
    {
        auto& curIterationState = mAtrousState.mIterationState[iteration];

        pRenderContext->clearUAV(curIterationState.pdaSigma->getUAV().get(), Falcor::uint4(0));
        pRenderContext->clearUAV(curIterationState.pdaKernel->getUAV().get(), Falcor::uint4(0));
        pRenderContext->clearUAV(curIterationState.pdaVarianceKernel->getUAV().get(), Falcor::uint4(0));
        pRenderContext->clearUAV(curIterationState.pdaLuminanceParams->getUAV().get(), Falcor::uint4(0));
        pRenderContext->clearUAV(curIterationState.pdaWeightFunctionParams->getUAV().get(), Falcor::uint4(0));
        pRenderContext->clearUAV(curIterationState.pdaIllumination->getUAV().get(), Falcor::uint4(0));

        perImageCB_D["daSigma"] = curIterationState.pdaSigma;
        perImageCB_D["daKernel"] = curIterationState.pdaKernel;
        perImageCB_D["daVarianceKernel"] = curIterationState.pdaVarianceKernel;
        perImageCB_D["daLuminanceParams"] = curIterationState.pdaLuminanceParams;
        perImageCB_D["daWeightFunctionParams"] = curIterationState.pdaWeightFunctionParams;

        perImageCB["dvSigmaL"] = curIterationState.dvSigmaL;
        perImageCB["dvSigmaZ"] = curIterationState.dvSigmaZ;
        perImageCB["dvSigmaN"] = curIterationState.dvSigmaN;

        perImageCB["dvLuminanceParams"] = curIterationState.dvLuminanceParams;

        for (int i = 0; i < 3; i++) {
            perImageCB["dvWeightFunctionParams"][i] = curIterationState.dvWeightFunctionParams[i];
        }

        for (int i = 0; i < 3; i++) {
            perImageCB["dvKernel"][i] = curIterationState.dvKernel[i];
        }

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                perImageCB["dvVarianceKernel"][i][j] = curIterationState.dvVarianceKernel[i][j];
            }
        }

        perImageCB_D["drIllumination"] = (iteration == mFilterIterations - 1 ? mFinalModulateState.pdaIllumination : mAtrousState.mIterationState[iteration + 1].pdaIllumination);
        perImageCB["daIllumination"] = curIterationState.pdaIllumination;

        perImageCB["gIllumination"] = curIterationState.pgIllumination;
        perImageCB["gStepSize"] = 1 << iteration;

        perImageCB_D["iteration"] = iteration;

        mAtrousState.dPass->execute(pRenderContext, mpDummyFullscreenFbo);
    }
}

void SVGFPass::computeFilteredMoments(RenderContext* pRenderContext)
{
    auto perImageCB = mpFilterMoments->getRootVar()["PerImageCB"];

    perImageCB["gIllumination"] = mpCurReprojFbo->getColorTexture(0);
    perImageCB["gHistoryLength"] = mpCurReprojFbo->getColorTexture(2);
    perImageCB["gLinearZAndNormal"] = mpLinearZAndNormalFbo->getColorTexture(0);
    perImageCB["gMoments"] = mpCurReprojFbo->getColorTexture(1);

    perImageCB["dvSigmaL"] = mFilterMomentsState.dvSigmaL;
    perImageCB["dvSigmaZ"] = mFilterMomentsState.dvSigmaZ;
    perImageCB["dvSigmaN"] = mFilterMomentsState.dvSigmaN;

    perImageCB["dvLuminanceParams"] =mFilterMomentsState. dvLuminanceParams;
    perImageCB["dvVarianceBoostFactor"] = mFilterMomentsState.dvVarianceBoostFactor;

    for (int i = 0; i < 3; i++) {
        perImageCB["dvWeightFunctionParams"][i] = mFilterMomentsState.dvWeightFunctionParams[i];
    }

    mpFilterMoments->execute(pRenderContext, mpPingPongFbo[0]);
}

void SVGFPass::computeDerivFilteredMoments(RenderContext* pRenderContext)
{
    auto perImageCB = mFilterMomentsState.dPass->getRootVar()["PerImageCB"];

    perImageCB["gIllumination"]     = mpCurReprojFbo->getColorTexture(0);
    perImageCB["gHistoryLength"]    = mpCurReprojFbo->getColorTexture(2);
    perImageCB["gLinearZAndNormal"]          = mpLinearZAndNormalFbo->getColorTexture(0);
    perImageCB["gMoments"]          = mpCurReprojFbo->getColorTexture(1);

    pRenderContext->clearUAV(mFilterMomentsState.pdaIllumination->getUAV().get(), Falcor::uint4(0));
    pRenderContext->clearUAV(mFilterMomentsState.pdaMoments->getUAV().get(), Falcor::uint4(0));

    perImageCB["daIllumination"]     = mFilterMomentsState.pdaIllumination;
    perImageCB["daHistoryLen"]    = mAtrousState.pdaHistoryLen;
    perImageCB["daMoments"]          = mFilterMomentsState.pdaMoments;

    perImageCB["dvSigmaL"] = mFilterMomentsState.dvSigmaL;
    perImageCB["dvSigmaZ"] = mFilterMomentsState.dvSigmaZ;
    perImageCB["dvSigmaN"] = mFilterMomentsState.dvSigmaN;

    perImageCB["dvLuminanceParams"] =mFilterMomentsState. dvLuminanceParams;
    perImageCB["dvVarianceBoostFactor"] = mFilterMomentsState.dvVarianceBoostFactor;

    for (int i = 0; i < 3; i++) {
        perImageCB["dvWeightFunctionParams"][i] = mFilterMomentsState.dvWeightFunctionParams[i];
    }

    auto perImageCB_D = mFilterMomentsState.dPass->getRootVar()["PerImageCB_D"];

    perImageCB_D["drIllumination"] = mAtrousState.mIterationState[0].pdaIllumination;
    perImageCB_D["daSigmaL"] = mFilterMomentsState.pdaSigmaL;
    perImageCB_D["daSigmaZ"] = mFilterMomentsState.pdaSigmaZ;
    perImageCB_D["daSigmaN"] = mFilterMomentsState.pdaSigmaN;

    perImageCB_D["daVarianceBoostFactor"] = mFilterMomentsState.pdaVarianceBoostFactor;
    perImageCB_D["daLuminanceParams"] = mFilterMomentsState.pdaLuminanceParams;
    perImageCB_D["daWeightFunctionParams"] = mFilterMomentsState.pdaWeightFunctionParams;

    mFilterMomentsState.dPass->execute(pRenderContext, mpDummyFullscreenFbo);
}

void SVGFPass::computeReprojection(RenderContext* pRenderContext, ref<Texture> pAlbedoTexture,
                                   ref<Texture> pColorTexture, ref<Texture> pEmissionTexture,
                                   ref<Texture> pMotionVectorTexture,
                                   ref<Texture> pPositionNormalFwidthTexture,
                                   ref<Texture> pPrevLinearZTexture,
                                   ref<Texture> pDebugTexture
    )
{
    auto perImageCB = mpReprojection->getRootVar()["PerImageCB"];

    // Setup textures for our reprojection shader pass
    perImageCB["gMotion"] = pMotionVectorTexture;
    perImageCB["gColor"] = pColorTexture;
    perImageCB["gEmission"] = pEmissionTexture;
    perImageCB["gAlbedo"] = pAlbedoTexture;
    perImageCB["gPositionNormalFwidth"] = pPositionNormalFwidthTexture;
    perImageCB["gPrevIllum"] = mpFilteredPastFbo->getColorTexture(0);
    perImageCB["gPrevMoments"] = mpPrevReprojFbo->getColorTexture(1);
    perImageCB["gLinearZAndNormal"] = mpLinearZAndNormalFbo->getColorTexture(0);
    perImageCB["gPrevLinearZAndNormal"] = pPrevLinearZTexture;
    perImageCB["gPrevHistoryLength"] = mpPrevReprojFbo->getColorTexture(2);

    // Setup variables for our reprojection pass
    perImageCB["dvAlpha"] = mReprojectState.dvAlpha;
    perImageCB["dvMomentsAlpha"] = mReprojectState.dvMomentsAlpha;

    perImageCB["dvLuminanceParams"] = mReprojectState.dvLuminanceParams;

    for (int i = 0; i < 3; i++) {
        perImageCB["dvReprojKernel"][i] = mReprojectState.dvKernel[i];
    }

    for (int i = 0; i < 4; i++) {
        perImageCB["dvReprojParams"][i] = mReprojectState.dvParams[i];
    }

    mpReprojection->execute(pRenderContext, mpCurReprojFbo);

    // save a copy of our past filtration for backwards differentiation
    pRenderContext->blit(mpFilteredPastFbo->getColorTexture(0)->getSRV(), mReprojectState.pPrevIllum->getRTV());
}

void SVGFPass::computeDerivReprojection(RenderContext* pRenderContext, ref<Texture> pAlbedoTexture,
                                   ref<Texture> pColorTexture, ref<Texture> pEmissionTexture,
                                   ref<Texture> pMotionVectorTexture,
                                   ref<Texture> pPositionNormalFwidthTexture,
                                   ref<Texture> pPrevLinearZTexture,
                                   ref<Texture> pDebugTexture
    )
{
    auto perImageCB = mReprojectState.dPass->getRootVar()["PerImageCB"];

    // Setup textures for our reprojection shader pass
    perImageCB["gMotion"]        = pMotionVectorTexture;
    perImageCB["gColor"]         = pColorTexture;
    perImageCB["gEmission"]      = pEmissionTexture;
    perImageCB["gAlbedo"]        = pAlbedoTexture;
    perImageCB["gPositionNormalFwidth"] = pPositionNormalFwidthTexture;
    perImageCB["gPrevIllum"]     = mReprojectState.pPrevIllum;
    perImageCB["gPrevMoments"]   = mpPrevReprojFbo->getColorTexture(1);
    perImageCB["gLinearZAndNormal"]       = mpLinearZAndNormalFbo->getColorTexture(0);
    perImageCB["gPrevLinearZAndNormal"]   = pPrevLinearZTexture;
    perImageCB["gPrevHistoryLength"] = mpPrevReprojFbo->getColorTexture(2);

    // Setup variables for our reprojection pass
    perImageCB["dvAlpha"] = mReprojectState.dvAlpha;
    perImageCB["dvMomentsAlpha"] = mReprojectState.dvMomentsAlpha;

    perImageCB["dvLuminanceParams"] = mReprojectState.dvLuminanceParams;

    for (int i = 0; i < 3; i++) {
        perImageCB["dvReprojKernel"][i] = mReprojectState.dvKernel[i];
    }

    for (int i = 0; i < 4; i++) {
        perImageCB["dvReprojParams"][i] = mReprojectState.dvParams[i];
    }

    auto perImageCB_D = mReprojectState.dPass->getRootVar()["PerImageCB_D"];

    perImageCB_D["drIllumination"] = mFilterMomentsState.pdaIllumination;
    perImageCB_D["drHistoryLen"] = mAtrousState.pdaHistoryLen;
    perImageCB_D["drMoments"] = mFilterMomentsState.pdaMoments;

    pRenderContext->clearUAV(mReprojectState.pdaLuminanceParams->getUAV().get(), Falcor::uint4(0));
    pRenderContext->clearUAV(mReprojectState.pdaReprojKernel->getUAV().get(), Falcor::uint4(0));
    pRenderContext->clearUAV(mReprojectState.pdaReprojParams->getUAV().get(), Falcor::uint4(0));
    pRenderContext->clearUAV(mReprojectState.pdaAlpha->getUAV().get(), Falcor::uint4(0));
    pRenderContext->clearUAV(mReprojectState.pdaMomentsAlpha->getUAV().get(), Falcor::uint4(0));

    perImageCB_D["daLuminanceParams"] = mReprojectState.pdaLuminanceParams;
    perImageCB_D["daReprojKernel"] = mReprojectState.pdaReprojKernel;
    perImageCB_D["daReprojParams"] = mReprojectState.pdaReprojParams;
    perImageCB_D["daAlpha"] = mReprojectState.pdaAlpha;
    perImageCB_D["daMomentsAlpha"] = mReprojectState.pdaMomentsAlpha;

    mReprojectState.dPass->execute(pRenderContext, mpDummyFullscreenFbo);
}


// Extracts linear z and its derivative from the linear Z texture and packs
// the normal from the world normal texture and packes them into the FBO.
// (It's slightly wasteful to copy linear z here, but having this all
// together in a single buffer is a small simplification, since we make a
// copy of it to refer to in the next frame.)
void SVGFPass::computeLinearZAndNormal(RenderContext* pRenderContext, ref<Texture> pLinearZTexture, ref<Texture> pWorldNormalTexture)
{
    auto perImageCB = mpPackLinearZAndNormal->getRootVar()["PerImageCB"];
    perImageCB["gLinearZ"] = pLinearZTexture;
    perImageCB["gNormal"] = pWorldNormalTexture;

    mpPackLinearZAndNormal->execute(pRenderContext, mpLinearZAndNormalFbo);
}

void SVGFPass::renderUI(Gui::Widgets& widget)
{
    float dummyVal;

    int dirty = 0;
    dirty |= (int)widget.checkbox("Enable SVGF", mFilterEnabled);

    widget.text("");
    widget.text("Number of filter iterations.  Which");
    widget.text("    iteration feeds into future frames?");
    dirty |= (int)widget.var("Iterations", mFilterIterations, 1, 10, 1);
    dirty |= (int)widget.var("Feedback", mFeedbackTap, -1, mFilterIterations - 2, 1);

    widget.var("mDI", mDerivativeInteration, 0, mFilterIterations - 1, 1);

    widget.text("");
    widget.text("Contol edge stopping on bilateral fitler");
    dirty |= (int)widget.var("For Color", dummyVal, 0.0f, 10000.0f, 0.01f);  // pass in sigma l as dummy var
    dirty |= (int)widget.var("For Normal", dummyVal, 0.001f, 1000.0f, 0.2f);

    widget.text("");
    widget.text("How much history should be used?");
    widget.text("    (alpha; 0 = full reuse; 1 = no reuse)");
    dirty |= (int)widget.var("Alpha", dummyVal, 0.0f, 1.0f, 0.001f);
    dirty |= (int)widget.var("Moments Alpha", dummyVal, 0.0f, 1.0f, 0.001f);

    if (dirty)
        mBuffersNeedClear = true;
}

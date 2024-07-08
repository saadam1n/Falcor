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

    const char kBufferShaderCompacting[]      = "RenderPasses/SVGFPass/SVGFBufferCompacting.ps.slang";
    const char kBufferShaderSumming[]         = "RenderPasses/SVGFPass/SVGFBufferSumming.cs.slang";
    const char kBufferShaderToTexture[]       = "RenderPasses/SVGFPass/SVGFBufferToTexture.ps.slang";

    const char kDerivativeVerifyShader[]      = "RenderPasses/SVGFPass/SVGFDerivativeVerify.ps.slang";

    const char kFilterMomentShaderS[]         = "RenderPasses/SVGFPass/SVGFFilterMomentsS.ps.slang";
    const char kFilterMomentShaderD[]         = "RenderPasses/SVGFPass/SVGFFilterMomentsD.ps.slang";

    const char kFinalModulateShaderS[]        = "RenderPasses/SVGFPass/SVGFFinalModulateS.ps.slang";
    const char kFinalModulateShaderD[]        = "RenderPasses/SVGFPass/SVGFFinalModulateD.ps.slang";

    const char kLossShader[]                  = "RenderPasses/SVGFPass/SVGFLoss.ps.slang";
    const char kLossGaussianShaderS[]         = "RenderPasses/SVGFPass/SVGFLossGaussianS.ps.slang";
    const char kLossGaussianShaderD[]         = "RenderPasses/SVGFPass/SVGFLossGaussianD.ps.slang";

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
    const char kOutputReference[] = "Reference";
    const char kOutputLoss[] = "Loss";

    // Stuff from dataset
    const std::string kDatasetReference = "Reference";
    const std::string kDatasetColor = "Color";
    const std::string kDatasetAlbedo = "Albedo";
    const std::string kDatasetEmission = "Emission";
    const std::string kDatasetWorldPosition = "WorldPosition";
    const std::string kDatasetWorldNormal = "WorldNormal";
    const std::string kDatasetPosNormalFwidth = "PositionNormalFwidth";
    const std::string kDatasetLinearZ = "LinearZ";
    const std::string kDatasetMotionVector = "MotionVec";

    // set common stuff first
    const size_t screenWidth = 1920;
    const size_t screenHeight = 1080;
    const size_t numPixels = screenWidth * screenHeight;

    const float3 dvLuminanceParams = float3(0.2126f, 0.7152f, 0.0722f);

    const float   dvSigmaL              = 10.0f;
    const float   dvSigmaZ              = 1.0;
    const float   dvSigmaN              = 128.0f;
    const float   dvAlpha               = 0.05f;
    const float   dvMomentsAlpha        = 0.2f;

    // x is L
    // y is Z
    // z is N
    const float3  dvSigma = float3(dvSigmaL, dvSigmaZ, dvSigmaN);

    const float dvWeightFunctionParams[3] {1.0, 1.0, 1.0};
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, SVGFPass>();
}


ref<Buffer> createAccumulationBuffer(ref<Device> pDevice, int bytes_per_elem = sizeof(float4), bool need_reaback = false) {
    return make_ref<Buffer>(pDevice, bytes_per_elem * numPixels, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, need_reaback ? MemoryType::ReadBack : MemoryType::DeviceLocal, nullptr);
}

ref<Texture> createFullscreenTexture(ref<Device> pDevice, ResourceFormat fmt = ResourceFormat::RGBA32Float)
{
    return make_ref<Texture>(pDevice, Resource::Type::Texture2D, fmt, screenWidth, screenHeight,  1, 1, 1, 1, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, nullptr);
}

SVGFRenderData::SVGFRenderData(const RenderData& renderData) {
    pAlbedoTexture = renderData.getTexture(kInputBufferAlbedo);
    pColorTexture = renderData.getTexture(kInputBufferColor);
    pEmissionTexture = renderData.getTexture(kInputBufferEmission);
    pWorldPositionTexture = renderData.getTexture(kInputBufferWorldPosition);
    pWorldNormalTexture = renderData.getTexture(kInputBufferWorldNormal);
    pPosNormalFwidthTexture = renderData.getTexture(kInputBufferPosNormalFwidth);
    pLinearZTexture = renderData.getTexture(kInputBufferLinearZ);
    pMotionVectorTexture = renderData.getTexture(kInputBufferMotionVector);
    pPrevLinearZAndNormalTexture = renderData.getTexture(kInternalBufferPreviousLinearZAndNormal);
    pOutputTexture = renderData.getTexture(kOutputBufferFilteredImage);
    pDebugTexture = renderData.getTexture(kOutputDebugBuffer);
    pDerivVerifyTexture = renderData.getTexture(kOutputDerivVerifyBuf);
    pLossTexture = renderData.getTexture(kOutputLoss);
    pReferenceTexture = renderData.getTexture(kOutputReference);
}

SVGFTrainingDataset::SVGFTrainingDataset(ref<Device> pDevice, const std::string& folder) : mFolder(folder), mSampleIdx(0) {
#define MarkDatasetTexture(x) p##x##Texture = createFullscreenTexture(pDevice); mTextureNameMappings[kDataset##x] = p##x##Texture;

    MarkDatasetTexture(Reference);
    MarkDatasetTexture(Albedo);
    MarkDatasetTexture(Color);
    MarkDatasetTexture(Emission);
    MarkDatasetTexture(WorldPosition);
    MarkDatasetTexture(WorldNormal);
    MarkDatasetTexture(PosNormalFwidth);
    MarkDatasetTexture(Reference);
    MarkDatasetTexture(LinearZ);
    MarkDatasetTexture(MotionVector);

    pLossTexture = createFullscreenTexture(pDevice);
    pPrevLinearZAndNormalTexture = createFullscreenTexture(pDevice);
    pOutputTexture = createFullscreenTexture(pDevice);
    pDebugTexture = createFullscreenTexture(pDevice);
    pDerivVerifyTexture = createFullscreenTexture(pDevice);

    // preload all textures
    std::map<std::string, std::future<Bitmap::UniqueConstPtr>> preloadTasks;
    while(atValidIndex())
    {
        for(auto [buffer, tex] : mTextureNameMappings)
        {
            std::string path = getSampleBufferPath(buffer);
            preloadTasks[path] = std::async(std::launch::async, &Bitmap::createFromFile, path, true, Falcor::Bitmap::ImportFlags::None);
        }

        mSampleIdx++;
    }

    mSampleIdx = 0;

    for(auto& [path, bitmap] : preloadTasks)
    {
        mPreloadedBitmaps[path] = std::move(bitmap.get());
    }
}

bool SVGFTrainingDataset::loadNext(RenderContext* pRenderContext)
{
    // check if we have more samples to read
    if (!atValidIndex())
    {
        mSampleIdx = 0;
        return false;
    }

    // continue with loading of samples
    for(auto [buffer, tex] : mTextureNameMappings)
    {
        loadSampleBuffer(pRenderContext, tex, buffer);
    }

    mSampleIdx++;

    // indicate to callee that we this sample was successfully read
    return true;
}

bool SVGFTrainingDataset::atValidIndex() const
{
    return std::filesystem::exists(getSampleBufferPath(kDatasetColor));
}

std::string SVGFTrainingDataset::getSampleBufferPath(const std::string& buffer) const
{
    return mFolder + std::to_string(mSampleIdx) + "-" + buffer + ".exr";
}

void SVGFTrainingDataset::loadSampleBuffer(RenderContext* pRenderContext, ref<Texture> tex, const std::string& buffer)
{
    std::string path = getSampleBufferPath(buffer);

    if(mPreloadedBitmaps.count(path) == 0)
    {
        mPreloadedBitmaps[path] = std::move(Bitmap::createFromFile(path, true));
    }

    if(pRenderContext) {
        pRenderContext->updateTextureData(tex.get(), (const void*)mPreloadedBitmaps[path]->getData());
    }
}

#define registerParameter(x) registerParameterUM(x, #x)

SVGFPass::SVGFPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice), mTrainingDataset(pDevice, "C:/FalcorFiles/Dataset0/")
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


    mPackLinearZAndNormalState.sPass = FullScreenPass::create(mpDevice, kPackLinearZAndNormalShader);

    mReprojectState.sPass = FullScreenPass::create(mpDevice, kReprojectShaderS);
    mReprojectState.dPass = FullScreenPass::create(mpDevice, kReprojectShaderD);

    mFilterMomentsState.sPass = FullScreenPass::create(mpDevice, kFilterMomentShaderS);
    mFilterMomentsState.dPass = FullScreenPass::create(mpDevice, kFilterMomentShaderD);

    mAtrousState.sPass = FullScreenPass::create(mpDevice, kAtrousShaderS);
    mAtrousState.dPass = FullScreenPass::create(mpDevice, kAtrousShaderD);

    mFinalModulateState.sPass = FullScreenPass::create(mpDevice, kFinalModulateShaderS);
    mFinalModulateState.dPass = FullScreenPass::create(mpDevice, kFinalModulateShaderD);

    compactingPass = FullScreenPass::create(mpDevice, kBufferShaderCompacting);
    summingPass = ComputePass::create(mpDevice, kBufferShaderSumming);
    bufferToTexturePass = FullScreenPass::create(mpDevice, kBufferShaderToTexture);

    mLossState.dPass = FullScreenPass::create(mpDevice, kLossShader);
    mLossState.sGaussianPass = FullScreenPass::create(mpDevice, kLossGaussianShaderS);
    mLossState.dGaussianPass = FullScreenPass::create(mpDevice, kLossGaussianShaderD);

    mpDerivativeVerify = FullScreenPass::create(mpDevice, kDerivativeVerifyShader);
    mpFuncOutputLower =  make_ref<Texture>(pDevice, Resource::Type::Texture2D, ResourceFormat::RGBA32Float, screenWidth, screenHeight,  1, 1, 1, 1, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource, nullptr);
    mpFuncOutputUpper =  make_ref<Texture>(pDevice, Resource::Type::Texture2D, ResourceFormat::RGBA32Float, screenWidth, screenHeight,  1, 1, 1, 1, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource, nullptr);

    // set reproj params
    mReprojectState.mLuminanceParams.dv = dvLuminanceParams;
    registerParameter(mReprojectState.mLuminanceParams);

    mReprojectState.mAlpha.dv = dvAlpha;
    registerParameter(mReprojectState.mAlpha);

    mReprojectState.mMomentsAlpha.dv = dvMomentsAlpha;
    registerParameter(mReprojectState.mMomentsAlpha);

    mReprojectState.mParams.dv[0] = 32.0;
    mReprojectState.mParams.dv[1] = 1.0;
    mReprojectState.mParams.dv[2] = 10.0;
    mReprojectState.mParams.dv[3] = 16.0;
    registerParameter(mReprojectState.mParams);

    mReprojectState.mKernel.dv[0] = 1.0;
    mReprojectState.mKernel.dv[1] = 1.0;
    mReprojectState.mKernel.dv[2] = 1.0;
    registerParameter(mReprojectState.mKernel);

    mReprojectState.pPrevFiltered = createFullscreenTexture(pDevice);

    // set filter moments params
    mFilterMomentsState.pdaHistoryLen = createAccumulationBuffer(pDevice);

    mFilterMomentsState.mSigma.dv = dvSigma;
    registerParameter(mFilterMomentsState.mSigma);

    mFilterMomentsState.mLuminanceParams.dv = dvLuminanceParams;
    registerParameter(mFilterMomentsState.mLuminanceParams);

    for (int i = 0; i < 3; i++) {
        mFilterMomentsState.mWeightFunctionParams.dv[i] = dvWeightFunctionParams[i];
    }
    registerParameter(mFilterMomentsState.mWeightFunctionParams);

    mFilterMomentsState.mVarianceBoostFactor.dv = 4.0;
    registerParameter(mFilterMomentsState.mVarianceBoostFactor);

    // Set atrous state vars

    mAtrousState.mIterationState.resize(mFilterIterations);
    for (auto& iterationState : mAtrousState.mIterationState)
    {
        iterationState.mSigma.dv = dvSigma;
        registerParameter(iterationState.mSigma);

        for (int i = 0; i < 3; i++) {
            iterationState.mWeightFunctionParams.dv[i] = dvWeightFunctionParams[i];
        }
        registerParameter(iterationState.mWeightFunctionParams);

        iterationState.mLuminanceParams.dv = dvLuminanceParams;
        registerParameter(iterationState.mLuminanceParams);

        iterationState.mKernel.dv[0] = 1.0;
        iterationState.mKernel.dv[1] = 2.0f / 3.0f;
        iterationState.mKernel.dv[2] = 1.0f / 6.0f;
        registerParameter(iterationState.mKernel);

        iterationState.mVarianceKernel.dv[0][0] = 1.0 / 4.0;
        iterationState.mVarianceKernel.dv[0][1] = 1.0 / 8.0;
        iterationState.mVarianceKernel.dv[1][0] = 1.0 / 8.0;
        iterationState.mVarianceKernel.dv[1][1] = 1.0 / 16.0;
        registerParameter(iterationState.mVarianceKernel);

        iterationState.pgIllumination = createFullscreenTexture(pDevice);
    }

    for (int i = 0; i < 2; i++)
    {
        pdaRawOutputBuffer[i] = createAccumulationBuffer(pDevice, sizeof(float4) * 50);
        pdaCompactedBuffer[i] = createAccumulationBuffer(pDevice);
    }

    for (int i = 0; i < 2; i++)
    {
        pdaPingPongSumBuffer[i] = createAccumulationBuffer(pDevice, sizeof(float4));
    }

    // set final modulate state vars
    mFinalModulateState.pdaIllumination = createAccumulationBuffer(pDevice);

    // set loss vars
    mLossState.pGaussianXInput = createFullscreenTexture(pDevice);
    mLossState.pGaussianYInput = createFullscreenTexture(pDevice);
    mLossState.pFilteredGaussian = createFullscreenTexture(pDevice);
    mLossState.pReferenceGaussian = createFullscreenTexture(pDevice);

    for (int i = 0; i < 3; i++)
    {
        mReadbackBuffer[i] = createAccumulationBuffer(pDevice, sizeof(float4), true);
    }

    mAtrousState.mSaveIllum = createFullscreenTexture(pDevice, ResourceFormat::RGBA32Int);

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

    reflector.addOutput(kOutputBufferFilteredImage, "Filtered image").format(ResourceFormat::RGBA16Float);
    reflector.addOutput(kOutputDebugBuffer, "DebugBuf").format(ResourceFormat::RGBA32Float);
    reflector.addOutput(kOutputDerivVerifyBuf, "Deriv Verify").format(ResourceFormat::RGBA32Float);
    reflector.addOutput(kOutputFuncLower, "Func lower").format(ResourceFormat::RGBA32Float);
    reflector.addOutput(kOutputFuncUpper, "Func upper").format(ResourceFormat::RGBA32Float);
    reflector.addOutput(kOutputFdCol, "FdCol").format(ResourceFormat::RGBA32Float);
    reflector.addOutput(kOutputBdCol, "BdCol").format(ResourceFormat::RGBA32Float);
    reflector.addOutput(kOutputReference, "Reference").format(ResourceFormat::RGBA32Float);
    reflector.addOutput(kOutputLoss, "Loss").format(ResourceFormat::RGBA32Float);

    return reflector;
}

void SVGFPass::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    allocateFbos(compileData.defaultTexDims, pRenderContext);
    mBuffersNeedClear = true;
}

void SVGFPass::runSvgfFilter(RenderContext* pRenderContext, const SVGFRenderData& renderData, bool updateInternalBuffers)
{
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
        computeLinearZAndNormal(pRenderContext, renderData.pLinearZTexture, renderData.pWorldNormalTexture);

        // Demodulate input color & albedo to get illumination and lerp in
        // reprojected filtered illumination from the previous frame.
        // Stores the result as well as initial moments and an updated
        // per-pixel history length in mpCurReprojFbo.

        computeReprojection(pRenderContext, renderData.pAlbedoTexture, renderData.pColorTexture, renderData.pEmissionTexture,
                            renderData.pMotionVectorTexture, renderData.pPosNormalFwidthTexture,
                            renderData.pPrevLinearZAndNormalTexture, renderData.pDebugTexture);

        // Do a first cross-bilateral filtering of the illumination and
        // estimate its variance, storing the result into a float4 in
        // mpPingPongFbo[0].  Takes mpCurReprojFbo as input.
        computeFilteredMoments(pRenderContext);

        //pRenderContext->blit(mpPingPongFbo[0]->getColorTexture(1)->getSRV(), pDebugTexture->getRTV());

        // Filter illumination from mpCurReprojFbo[0], storing the result
        // in mpPingPongFbo[0].  Along the way (or at the end, depending on
        // the value of mFeedbackTap), save the filtered illumination for
        // next time into mpFilteredPastFbo.
        computeAtrousDecomposition(pRenderContext, renderData.pAlbedoTexture, updateInternalBuffers);

        // Compute albedo * filtered illumination and add emission back in.
        auto perImageCB = mFinalModulateState.sPass->getRootVar()["PerImageCB"];
        perImageCB["gAlbedo"] = renderData.pAlbedoTexture;
        perImageCB["gEmission"] = renderData.pEmissionTexture;
        perImageCB["gIllumination"] = mpPingPongFbo[0]->getColorTexture(0);
        mFinalModulateState.sPass->execute(pRenderContext, mpFinalFbo);

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
    runTrainingAndTesting(pRenderContext, renderData);
    std::cout.flush();
}

void SVGFPass::runTrainingAndTesting(RenderContext* pRenderContext, const RenderData& renderData)
{
    SVGFRenderData svgfrd(renderData);

    if (!mTrained)
    {
        runNextTrainingTask(pRenderContext);
        // display current results to screen
        pRenderContext->blit(mTrainingDataset.pOutputTexture->getSRV(), svgfrd.pOutputTexture->getRTV());
        pRenderContext->blit(mTrainingDataset.pLossTexture->getSRV(), svgfrd.pLossTexture->getRTV());
    }
    else
    {
        if(!pScene) return;

        runSvgfFilter(pRenderContext, svgfrd, true);

        // compute loss so we can see it on the screen
        pRenderContext->blit(mTrainingDataset.pReferenceTexture->getSRV(), svgfrd.pReferenceTexture->getRTV());
        computeLoss(pRenderContext, svgfrd);

        float4 loss;
        mpParallelReduction->execute(pRenderContext, svgfrd.pLossTexture, ParallelReduction::Type::Sum, &loss);

        // wait for all pending actions to execute
        pRenderContext->submit(true);
        std::cout << "Total loss: " << loss.x << "\n";
    }
}

void SVGFPass::runNextTrainingTask(RenderContext* pRenderContext)
{
    const int K_NUM_EPOCHS = 7;

    if(mEpoch < K_NUM_EPOCHS)
    {
        if (mDatasetIndex == 0) {
            mBuffersNeedClear = true;

            std::cout << "Running epoch\t" << mEpoch << "\n";
        }

        if (mTrainingDataset.loadNext(pRenderContext)) {
            std::cout << "\tOn Frame " << mDatasetIndex << "\n";
            // first clear all our buffers
            for (int i = 0; i < mParameterReflector.size(); i++)
            {
                mParameterReflector[i].mAddress->clearBuffer(pRenderContext);
            }

            runSvgfFilter(pRenderContext, mTrainingDataset, true);
            computeDerivatives(pRenderContext, mTrainingDataset, true);

            // now accumulate everything
            int baseOffset = mDatasetIndex * mParameterReflector.size() * sizeof(float4);
            for (int i = 0; i < mParameterReflector.size(); i++)
            {
                int offset = i * sizeof(float4);
                reduceParameter(pRenderContext, *mParameterReflector[i].mAddress, baseOffset + offset);
            }

            mpParallelReduction->execute<float4>(pRenderContext, mTrainingDataset.pLossTexture, ParallelReduction::Type::Sum, nullptr, mReadbackBuffer[2], mDatasetIndex * sizeof(float4));

            // keep a copy of our output
            //mTrainingDataset.pOutputTexture->captureToFile(0, 0, "C:/FalcorFiles/TrainingDump/" + std::to_string(epoch) + ".exr", Falcor::Bitmap::FileFormat::ExrFile, Falcor::Bitmap::ExportFlags::None, false);

            mDatasetIndex++;
        }
        else
        {
            int batchSize = mDatasetIndex;

            const float K_LRATE_NUMER = 11.5f * 20.0f * 20.0f;
            const float K_LRATE_DENOM = 1.0f * 20.0f;

            // skip the first few frames which probably don't have stablized derivatives
            const int K_FRAME_SAMPLE_START = 6;
            int sampledFrames = batchSize - K_FRAME_SAMPLE_START;

            float learningRate = K_LRATE_NUMER / ((K_LRATE_DENOM + mEpoch) * sampledFrames);



            // adjust values
            float max_adjustment = 0.0f;
            {
                float4* gradient = (float4*)mReadbackBuffer[0]->map();

                for (int i = 0; i < mParameterReflector.size(); i++)
                {
                    auto& pmi = mParameterReflector[i];

                    float4 totalGradient = float4(0.0f);
                    for(int j = K_FRAME_SAMPLE_START; j < batchSize; j++)
                    {
                        totalGradient += gradient[j * mParameterReflector.size() + i];
                    }

                    for (int j = 0; j < pmi.mNumElements; j++)
                    {
                        float adjustment = -learningRate * totalGradient[j];
                        std::cout << "\tAdjusting by " << pmi.name << "\tby " << adjustment << "\n";
                        pmi.mAddress->dv[j] += adjustment;

                        // ensure greater than zero
                        if (pmi.mAddress->dv[j] < 0.0f)
                        {
                            pmi.mAddress->dv[j] = 0.0f;
                        }

                        max_adjustment = std::max(max_adjustment, std::abs(adjustment));
                    }
                }

                mReadbackBuffer[0]->unmap();
            }
            std::cout << "Max adjustment was " << max_adjustment << "\n";

            // now wait for it to execute and download it
            float4 loss = float4(0.0f);
            {
                float4* perFrameLoss = (float4*)mReadbackBuffer[2]->map();

                for(int i = K_FRAME_SAMPLE_START; i < batchSize; i++)
                {
                    loss += perFrameLoss[i];
                }

                mReadbackBuffer[2]->unmap();
            }
            loss /= float4(sampledFrames);
            std::cout << "Total loss in epoch\t" << mEpoch << "\tacross " << sampledFrames << "\t frames was " << loss.r << "\n";

            std::cout << "\n\n\n\n\n\n";

            mDatasetIndex = 0;
            mEpoch++;
        }
    }
    else
    {
        mTrained = true;
    }
}


void SVGFPass::runDerivativeTest(RenderContext* pRenderContext, const RenderData& renderData)
{
    if(!pScene) return;

    SVGFRenderData svgfrd(renderData);

    mDelta = 0.05f;

    //float& valToChange = mReprojectState.mAlpha.dv;
    float& valToChange = mAtrousState.mIterationState[0].mSigma.dv.x;
    float oldval = valToChange;

    valToChange = oldval - mDelta;
    runSvgfFilter(pRenderContext, svgfrd, false);
    pRenderContext->blit(mpFinalFbo->getColorTexture(0)->getSRV(), mpFuncOutputLower->getRTV());
    pRenderContext->blit(mpFinalFbo->getColorTexture(0)->getSRV(), renderData.getTexture(kOutputFuncLower)->getRTV());


    valToChange = oldval + mDelta;
    runSvgfFilter(pRenderContext, svgfrd, false);
    pRenderContext->blit(mpFinalFbo->getColorTexture(0)->getSRV(), mpFuncOutputUpper->getRTV());
    pRenderContext->blit(mpFinalFbo->getColorTexture(0)->getSRV(),  renderData.getTexture(kOutputFuncUpper)->getRTV());
     
    valToChange = oldval;
     
    runSvgfFilter(pRenderContext, svgfrd, true);
    computeDerivatives(pRenderContext, svgfrd, false);
    computeDerivVerification(pRenderContext, svgfrd);
    pRenderContext->blit(mpDerivativeVerifyFbo->getColorTexture(1)->getSRV(),  renderData.getTexture(kOutputFdCol)->getRTV());
    pRenderContext->blit(mpDerivativeVerifyFbo->getColorTexture(2)->getSRV(),  renderData.getTexture(kOutputBdCol)->getRTV());

    pRenderContext->blit(mAtrousState.mSaveIllum->getSRV(),   renderData.getTexture(kOutputDebugBuffer)->getRTV());

    std::cout << "Fwd Diff Sum:\t" << getTexSum(pRenderContext, mpDerivativeVerifyFbo->getColorTexture(1)) << std::endl;
    std::cout << "Bwd Diff Sum:\t" << getTexSum(pRenderContext, mpDerivativeVerifyFbo->getColorTexture(2)) << std::endl;

    float4 falcorTest = float4(0.0f);

    mpParallelReduction->execute(pRenderContext, mpDerivativeVerifyFbo->getColorTexture(2), ParallelReduction::Type::Sum, &falcorTest, mReadbackBuffer[1]);

    pRenderContext->submit(true);

    std::cout << "Flr Redc Sum:\t" << falcorTest.x << std::endl;

#if 0
    int numRemaining = numPixels;
    for (int i = 0; i < 3; i++)
    {
        // clear the output buffer
        pRenderContext->clearUAV(pdaPingPongSumBuffer[1]->getUAV().get(), Falcor::uint4(0));

        auto summingCB = summingPass->getRootVar()["SummingCB"];

        summingCB["srcBuf"] = (i == 0 ? mFilterMomentsState.mSigma.da : pdaPingPongSumBuffer[0]);
        summingCB["srcOffset"] = 0;
        summingCB["srcMax"] = numRemaining;

        summingCB["dstBuf"] = pdaPingPongSumBuffer[1];
        summingCB["dstOffset"] = 0;

        summingPass->execute(pRenderContext, numRemaining, 1);
        // round up divide
        numRemaining = (numRemaining + 127) / 128;

        std::swap(pdaPingPongSumBuffer[0], pdaPingPongSumBuffer[1]);
    }

    // written results will be the first index
    pRenderContext->copyBufferRegion(mReadbackBuffer[0].get(), 0, pdaPingPongSumBuffer[0].get(), 0, sizeof(float4) * 1);
    float4* ptr = (float4*)mReadbackBuffer[0]->map();
    std::cout << "Par Redc Sum:\t" << ptr[0].x << std::endl;

    mReadbackBuffer[0]->unmap();
#endif

    std::cout << std::endl;
}

void SVGFPass::computeDerivVerification(RenderContext* pRenderContext, const SVGFRenderData& renderData)
{
    auto perImageCB = mpDerivativeVerify->getRootVar()["PerImageCB"];

    perImageCB["drBackwardsDiffBuffer"] = mAtrousState.mIterationState[0].mSigma.da;
    perImageCB["gFuncOutputLower"] = mpFuncOutputLower;
    perImageCB["gFuncOutputUpper"] = mpFuncOutputUpper;
    perImageCB["delta"] = mDelta;

    mpDerivativeVerify->execute(pRenderContext, mpDerivativeVerifyFbo);
    pRenderContext->blit(mpDerivativeVerifyFbo->getColorTexture(0)->getSRV(), renderData.pDerivVerifyTexture->getRTV());
}


// I'll move parts of this off to other function as need be
void SVGFPass::computeDerivatives(RenderContext* pRenderContext, const SVGFRenderData& renderData, bool useLoss)
{
    ref<Texture> pIllumTexture = mpPingPongFbo[0]->getColorTexture(0);


    if (mFilterEnabled) {
        if (useLoss)
        {
            computeLoss(pRenderContext, renderData);
        }
        else
        {
            // set everything to 1.0 (except the alpha channel)
            float4 defaultDerivative = float4(1.0, 1.0, 1.0, 0.0);
            uint4* dPtr = (uint4*)&defaultDerivative;
            pRenderContext->clearUAV(pdaCompactedBuffer[1]->getUAV().get(), *dPtr);
        }

        computeDerivFinalModulate(pRenderContext, renderData.pOutputTexture, pIllumTexture, renderData.pAlbedoTexture, renderData.pEmissionTexture);

        // now, the derivative is stored in mFinalModulateState.pdaIllum
        // now, we will have to computer the atrous decomposition reverse
        // the atrous decomp has multiple stages
        // each stage outputs the exact same result - a color
        // we need to use that color and its derivative to feed the previous pass
        // ideally, we will want a buffer and specific variables for each stage
        // right now, I'll just set up the buffers

        computeDerivAtrousDecomposition(pRenderContext, renderData.pAlbedoTexture, renderData.pOutputTexture);

        computeDerivFilteredMoments(pRenderContext);

        computeDerivReprojection(pRenderContext, renderData.pAlbedoTexture, renderData.pColorTexture, renderData.pEmissionTexture, renderData.pMotionVectorTexture, renderData.pPosNormalFwidthTexture, renderData.pLinearZTexture, renderData.pDebugTexture);
    }
}

void SVGFPass::computeLoss(RenderContext* pRenderContext, const SVGFRenderData& renderData)
{
    computeGaussian(pRenderContext, renderData.pReferenceTexture, mLossState.pReferenceGaussian, false);
    computeGaussian(pRenderContext, renderData.pOutputTexture, mLossState.pFilteredGaussian, true);

    clearRawOutputBuffer(pRenderContext, 0);
    clearRawOutputBuffer(pRenderContext, 1);

    auto perImageCB = mLossState.dPass->getRootVar()["PerImageCB"];

    perImageCB["filteredGaussian"] = mLossState.pFilteredGaussian;
    perImageCB["referenceGaussian"] = mLossState.pReferenceGaussian;

    perImageCB["filteredImage"] = renderData.pOutputTexture;
    perImageCB["referenceImage"] = renderData.pReferenceTexture;

    perImageCB["pdaFilteredGaussian"] = pdaRawOutputBuffer[0];
    perImageCB["pdaFilteredImage"] = pdaRawOutputBuffer[1];

    mLossState.dPass->execute(pRenderContext, mpDummyFullscreenFbo);

    pRenderContext->blit(mpDummyFullscreenFbo->getColorTexture(0)->getSRV(), renderData.pLossTexture->getRTV());

    runCompactingPass(pRenderContext, 0, 9);

    computeDerivGaussian(pRenderContext);
}

void SVGFPass::computeGaussian(RenderContext* pRenderContext, ref<Texture> tex, ref<Texture> storageLocation, bool saveTextures)
{
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
    auto perImageCB = mLossState.dGaussianPass->getRootVar()["PerImageCB"];
    auto perImageCB_D = mLossState.dGaussianPass->getRootVar()["PerImageCB_D"];

    clearRawOutputBuffer(pRenderContext, 0);
    perImageCB_D["drIllumination"] = pdaCompactedBuffer[0];

    perImageCB["image"] = mLossState.pGaussianYInput;
    perImageCB["yaxis"] = true;
    perImageCB["pdaIllumination"] = pdaRawOutputBuffer[0];
    mLossState.dGaussianPass->execute(pRenderContext, mpDummyFullscreenFbo);

    runCompactingPass(pRenderContext, 0, 11);

    perImageCB["image"] = mLossState.pGaussianXInput;
    perImageCB["yaxis"] = false;
    perImageCB["pdaIllumination"] = pdaRawOutputBuffer[1];
    mLossState.dGaussianPass->execute(pRenderContext, mpDummyFullscreenFbo);

    // we have the extra derivative from the loss pass
    // not a great way to encapsulate stuff but whatever
    runCompactingPass(pRenderContext, 1, 12);
}

void SVGFPass::computeDerivFinalModulate(RenderContext* pRenderContext, ref<Texture> pOutputTexture, ref<Texture> pIllumination, ref<Texture> pAlbedoTexture, ref<Texture> pEmissionTexture) {
    pRenderContext->clearUAV(mFinalModulateState.pdaIllumination->getUAV().get(), Falcor::uint4(0));

    auto perImageCB = mFinalModulateState.dPass->getRootVar()["PerImageCB"];
    perImageCB["gAlbedo"] = pAlbedoTexture;
    perImageCB["gEmission"] = pEmissionTexture;
    perImageCB["gIllumination"] = mpPingPongFbo[0]->getColorTexture(0);
    perImageCB["daIllumination"] = mFinalModulateState.pdaIllumination;

    auto perImageCB_D = mFinalModulateState.dPass->getRootVar()["PerImageCB_D"];
    perImageCB_D["drFilteredImage"] = pdaCompactedBuffer[1];

    mFinalModulateState.dPass->execute(pRenderContext, mpDerivativeVerifyFbo);
}

void SVGFPass::computeAtrousDecomposition(RenderContext* pRenderContext, ref<Texture> pAlbedoTexture, bool updateInternalBuffers)
{
    auto perImageCB = mAtrousState.sPass->getRootVar()["PerImageCB"];

    perImageCB["gAlbedo"] = pAlbedoTexture;
    perImageCB["gLinearZAndNormal"] = mpLinearZAndNormalFbo->getColorTexture(0);

    for (int iteration = 0; iteration < mFilterIterations; iteration++)
    {

        auto& curIterationState = mAtrousState.mIterationState[iteration];

        perImageCB["dvSigmaL"] = curIterationState.mSigma.dv.x;
        perImageCB["dvSigmaZ"] = curIterationState.mSigma.dv.y;
        perImageCB["dvSigmaN"] = curIterationState.mSigma.dv.z; 

        perImageCB["dvLuminanceParams"] = curIterationState.mLuminanceParams.dv;

        for (int i = 0; i < 3; i++) {
            perImageCB["dvWeightFunctionParams"][i] = curIterationState.mWeightFunctionParams.dv[i]; 
        }

        for (int i = 0; i < 3; i++) {
            perImageCB["dvKernel"][i] = curIterationState.mKernel.dv[i];
        }

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                perImageCB["dvVarianceKernel"][i][j] = curIterationState.mVarianceKernel.dv[i][j];
            }
        }


        ref<Fbo> curTargetFbo = mpPingPongFbo[1];
        // keep a copy of our input for backwards differation
        pRenderContext->blit(mpPingPongFbo[0]->getColorTexture(0)->getSRV(), curIterationState.pgIllumination->getRTV());

        perImageCB["gIllumination"] = mpPingPongFbo[0]->getColorTexture(0);
        perImageCB["gStepSize"] = 1 << iteration;


        mAtrousState.sPass->execute(pRenderContext, curTargetFbo);

        // store the filtered color for the feedback path
        if (updateInternalBuffers && iteration == std::min(mFeedbackTap, mFilterIterations - 1))
        {
            pRenderContext->blit(curTargetFbo->getColorTexture(0)->getSRV(), mpFilteredPastFbo->getRenderTargetView(0));
        }

        std::swap(mpPingPongFbo[0], mpPingPongFbo[1]);
    }

    if (updateInternalBuffers && mFeedbackTap < 0)
    {
        pRenderContext->blit(mpCurReprojFbo->getColorTexture(0)->getSRV(), mpFilteredPastFbo->getRenderTargetView(0));
    }
}

void SVGFPass::computeDerivAtrousDecomposition(RenderContext* pRenderContext, ref<Texture> pAlbedoTexture, ref<Texture> pOutputTexture)
{
    auto perImageCB = mAtrousState.dPass->getRootVar()["PerImageCB"];
    auto perImageCB_D = mAtrousState.dPass->getRootVar()["PerImageCB_D"];

    perImageCB["gAlbedo"]        = pAlbedoTexture;
    perImageCB["gLinearZAndNormal"]       = mpLinearZAndNormalFbo->getColorTexture(0);

    for (int iteration = mFilterIterations - 1; iteration >= 0; iteration--)
    {
        auto& curIterationState = mAtrousState.mIterationState[iteration];

        // clear raw output
        pRenderContext->clearUAV(pdaRawOutputBuffer[0]->getUAV().get(), Falcor::uint4(0));

        perImageCB_D["daSigma"] = curIterationState.mSigma.da;
        perImageCB_D["daKernel"] = curIterationState.mKernel.da;
        perImageCB_D["daVarianceKernel"] = curIterationState.mVarianceKernel.da;
        perImageCB_D["daLuminanceParams"] = curIterationState.mLuminanceParams.da;
        perImageCB_D["daWeightFunctionParams"] = curIterationState.mWeightFunctionParams.da;

        perImageCB["dvSigmaL"] = curIterationState.mSigma.dv.x;
        perImageCB["dvSigmaZ"] = curIterationState.mSigma.dv.y;
        perImageCB["dvSigmaN"] = curIterationState.mSigma.dv.z;

        perImageCB["dvLuminanceParams"] = curIterationState.mLuminanceParams.dv;

        for (int i = 0; i < 3; i++) {
            perImageCB["dvWeightFunctionParams"][i] = curIterationState.mWeightFunctionParams.dv[i];
        }

        for (int i = 0; i < 3; i++) {
            perImageCB["dvKernel"][i] = curIterationState.mKernel.dv[i];
        }

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                perImageCB["dvVarianceKernel"][i][j] = curIterationState.mVarianceKernel.dv[i][j];
            }
        }

        perImageCB_D["drIllumination"] = (iteration == mFilterIterations - 1 ? mFinalModulateState.pdaIllumination : pdaCompactedBuffer[0]);
        perImageCB["daIllumination"] = pdaRawOutputBuffer[0];

        perImageCB["gIllumination"] = curIterationState.pgIllumination;
        perImageCB["gStepSize"] = 1 << iteration;

        mAtrousState.dPass->execute(pRenderContext, mpDummyFullscreenFbo);

        runCompactingPass(pRenderContext, 0, 1 + 9 + 25);

    }
}

void SVGFPass::computeFilteredMoments(RenderContext* pRenderContext)
{
    auto perImageCB = mFilterMomentsState.sPass->getRootVar()["PerImageCB"];

    perImageCB["gIllumination"] = mpCurReprojFbo->getColorTexture(0);
    perImageCB["gHistoryLength"] = mpCurReprojFbo->getColorTexture(2);
    perImageCB["gLinearZAndNormal"] = mpLinearZAndNormalFbo->getColorTexture(0);
    perImageCB["gMoments"] = mpCurReprojFbo->getColorTexture(1);

    perImageCB["dvSigmaL"] = mFilterMomentsState.mSigma.dv.x;
    perImageCB["dvSigmaZ"] = mFilterMomentsState.mSigma.dv.y;
    perImageCB["dvSigmaN"] = mFilterMomentsState.mSigma.dv.z;

    perImageCB["dvLuminanceParams"] = mFilterMomentsState. mLuminanceParams.dv;
    perImageCB["dvVarianceBoostFactor"] = mFilterMomentsState.mVarianceBoostFactor.dv;

    for (int i = 0; i < 3; i++) {
        perImageCB["dvWeightFunctionParams"][i] = mFilterMomentsState.mWeightFunctionParams.dv[i];
    }

    mFilterMomentsState.sPass->execute(pRenderContext, mpPingPongFbo[0]);
}

void SVGFPass::computeDerivFilteredMoments(RenderContext* pRenderContext)
{
    auto perImageCB = mFilterMomentsState.dPass->getRootVar()["PerImageCB"];

    perImageCB["gIllumination"]     = mpCurReprojFbo->getColorTexture(0);
    perImageCB["gHistoryLength"]    = mpCurReprojFbo->getColorTexture(2);
    perImageCB["gLinearZAndNormal"]          = mpLinearZAndNormalFbo->getColorTexture(0);
    perImageCB["gMoments"]          = mpCurReprojFbo->getColorTexture(1);

    pRenderContext->clearUAV(pdaRawOutputBuffer[0]->getUAV().get(), Falcor::uint4(0));
    pRenderContext->clearUAV(pdaRawOutputBuffer[1]->getUAV().get(), Falcor::uint4(0));

    perImageCB["daIllumination"]     = pdaRawOutputBuffer[0];
    perImageCB["daMoments"]          = pdaRawOutputBuffer[1];
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

    perImageCB_D["drIllumination"] = pdaCompactedBuffer[0];
    perImageCB_D["daSigma"] = mFilterMomentsState.mSigma.da;

    perImageCB_D["daVarianceBoostFactor"] = mFilterMomentsState.mVarianceBoostFactor.da;
    perImageCB_D["daLuminanceParams"] = mFilterMomentsState.mLuminanceParams.da;
    perImageCB_D["daWeightFunctionParams"] = mFilterMomentsState.mWeightFunctionParams.da;

    mFilterMomentsState.dPass->execute(pRenderContext, mpDummyFullscreenFbo);

    runCompactingPass(pRenderContext, 0, 50);
    runCompactingPass(pRenderContext, 1, 49);
}

void SVGFPass::computeReprojection(RenderContext* pRenderContext, ref<Texture> pAlbedoTexture,
                                   ref<Texture> pColorTexture, ref<Texture> pEmissionTexture,
                                   ref<Texture> pMotionVectorTexture,
                                   ref<Texture> pPositionNormalFwidthTexture,
                                   ref<Texture> pPrevLinearZTexture,
                                   ref<Texture> pDebugTexture
    )
{
    auto perImageCB = mReprojectState.sPass->getRootVar()["PerImageCB"];

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
    perImageCB["dvAlpha"] = mReprojectState.mAlpha.dv;
    perImageCB["dvMomentsAlpha"] = mReprojectState.mMomentsAlpha.dv;

    perImageCB["dvLuminanceParams"] = mReprojectState.mLuminanceParams.dv;

    for (int i = 0; i < 3; i++) {
        perImageCB["dvReprojKernel"][i] = mReprojectState.mKernel.dv[i];
    }

    for (int i = 0; i < 4; i++) {
        perImageCB["dvReprojParams"][i] = mReprojectState.mParams.dv[i];
    }

    mReprojectState.sPass->execute(pRenderContext, mpCurReprojFbo);

    // save a copy of our past filtration for backwards differentiation
    pRenderContext->blit(mpFilteredPastFbo->getColorTexture(0)->getSRV(), mReprojectState.pPrevFiltered->getRTV());
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
    perImageCB["gPrevIllum"]     = mReprojectState.pPrevFiltered;
    perImageCB["gPrevMoments"]   = mpPrevReprojFbo->getColorTexture(1);
    perImageCB["gLinearZAndNormal"]       = mpLinearZAndNormalFbo->getColorTexture(0);
    perImageCB["gPrevLinearZAndNormal"]   = pPrevLinearZTexture;
    perImageCB["gPrevHistoryLength"] = mpPrevReprojFbo->getColorTexture(2);

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

    auto perImageCB_D = mReprojectState.dPass->getRootVar()["PerImageCB_D"];

    perImageCB_D["drIllumination"] = pdaCompactedBuffer[0];
    perImageCB_D["drMoments"] = pdaCompactedBuffer[1];
    perImageCB_D["drHistoryLen"] = mFilterMomentsState.pdaHistoryLen;

    perImageCB_D["daLuminanceParams"] = mReprojectState.mLuminanceParams.da;
    perImageCB_D["daReprojKernel"] = mReprojectState.mKernel.da;
    perImageCB_D["daReprojParams"] = mReprojectState.mParams.da;
    perImageCB_D["daAlpha"] = mReprojectState.mAlpha.da;
    perImageCB_D["daMomentsAlpha"] = mReprojectState.mMomentsAlpha.da;

    mReprojectState.dPass->execute(pRenderContext, mpDummyFullscreenFbo);
}


void SVGFPass::runCompactingPass(RenderContext* pRenderContext, int idx, int n)
{
    auto compactingCB = compactingPass->getRootVar()["CompactingCB"];
    compactingCB["drIllumination"] = pdaRawOutputBuffer[idx];
    compactingCB["daIllumination"] = pdaCompactedBuffer[idx];
    compactingCB["gAlbedo"] = mpDummyFullscreenFbo->getColorTexture(0);

    compactingCB["elements"] = n;
    // compact the raw output
    compactingPass->execute(pRenderContext, mpDummyFullscreenFbo);
}

void SVGFPass::clearRawOutputBuffer(RenderContext* pRenderContext, int idx)
{
    pRenderContext->clearUAV(pdaRawOutputBuffer[idx]->getUAV().get(), uint4(0));
}


#define USE_BUILTIN_PARALLEL_REDUCTION
void SVGFPass::reduceParameter(RenderContext* pRenderContext, SVGFParameter<float4>& param, int offset)
{
#ifdef USE_BUILTIN_PARALLEL_REDUCTION
    auto conversionCB = bufferToTexturePass->getRootVar()["ConversionCB"];

    conversionCB["drIllumination"] = param.da;

    bufferToTexturePass->execute(pRenderContext, bufferToTextureFbo);
    mpParallelReduction->execute<float4>(pRenderContext, bufferToTextureFbo->getColorTexture(0), ParallelReduction::Type::Sum, nullptr, mReadbackBuffer[0], offset);
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


void SVGFPass::registerParameterManual(SVGFParameter<float4>* param, int cnt, const std::string& name)
{
    param->da = createAccumulationBuffer(mpDevice);

    ParameterMetaInfo pmi{param, cnt, name};
    mParameterReflector.push_back(pmi);
}

// Extracts linear z and its derivative from the linear Z texture and packs
// the normal from the world normal texture and packes them into the FBO.
// (It's slightly wasteful to copy linear z here, but having this all
// together in a single buffer is a small simplification, since we make a
// copy of it to refer to in the next frame.)
void SVGFPass::computeLinearZAndNormal(RenderContext* pRenderContext, ref<Texture> pLinearZTexture, ref<Texture> pWorldNormalTexture)
{
    auto perImageCB = mPackLinearZAndNormalState.sPass->getRootVar()["PerImageCB"];
    perImageCB["gLinearZ"] = pLinearZTexture;
    perImageCB["gNormal"] = pWorldNormalTexture;

    mPackLinearZAndNormalState.sPass->execute(pRenderContext, mpLinearZAndNormalFbo);
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

    if (dirty)
        mBuffersNeedClear = true;
}

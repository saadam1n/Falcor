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
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Core/Pass/FullScreenPass.h"
#include "Utils/Algorithm/ParallelReduction.h"

#include "SVGFCommon.h"

using namespace Falcor;

class SVGFPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(SVGFPass, "SVGFPass", "SVGF denoising pass.");

    static ref<SVGFPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<SVGFPass>(pDevice, props); }

    SVGFPass(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void renderUI(Gui::Widgets& widget) override;

    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pSceneUpdate) override
    {
        this->pScene = pSceneUpdate;
    }

private:
    ref<SVGFUtilitySet> mpUtilities;

    ref<Scene> pScene;

    bool mTrained = false;
    int mEpoch = 0;
    void runNextTrainingTask(RenderContext* pRenderContext);

    void runDerivativeTest(RenderContext* pRenderContext, const RenderData& renderData);
    void runTrainingAndTesting(RenderContext* pRenderContext, const RenderData& renderData);

    void allocateFbos(uint2 dim, RenderContext* pRenderContext);
    void clearBuffers(RenderContext* pRenderContext, const SVGFRenderData& renderData);

    void computeLinearZAndNormal(RenderContext* pRenderContext, ref<Texture> pLinearZTexture,
                                 ref<Texture> pWorldNormalTexture);
    void computeReprojection(RenderContext* pRenderContext, ref<Texture> pAlbedoTexture,
                             ref<Texture> pColorTexture, ref<Texture> pEmissionTexture,
                             ref<Texture> pMotionVectorTexture,
                             ref<Texture> pPositionNormalFwidthTexture,
                             ref<Texture> pPrevLinearZAndNormalTexture,
                             ref<Texture> pDebugTexture
        );


    void computeFilteredMoments(RenderContext* pRenderContext);
    void computeAtrousDecomposition(RenderContext* pRenderContext, ref<Texture> pAlbedoTexture, bool updateInternalBuffers);
    void computeGaussian(RenderContext* pRenderContext, ref<Texture> tex, ref<Texture> storageLocation, bool saveTextures);

    void runSvgfFilter(RenderContext* pRenderContext, const SVGFRenderData& renderData, bool updateInternalBuffers);
    void computeDerivatives(RenderContext* pRenderContext, const SVGFRenderData& renderData, bool useLoss);
    void computeLoss(RenderContext* pRenderContext, const SVGFRenderData& renderData);
    void computeDerivGaussian(RenderContext* pRenderContext);
    void computeDerivFinalModulate(RenderContext* pRenderContext, ref<Texture> pResultantImage, ref<Texture> pIllumination, ref<Texture> pAlbedoTexture, ref<Texture> pEmissionTexture);
    void computeDerivAtrousDecomposition(RenderContext* pRenderContext, ref<Texture> pAlbedoTexture, ref<Texture> pOutputTexture);
    void computeDerivFilteredMoments(RenderContext* pRenderContext);
    void computeDerivReprojection(RenderContext* pRenderContext, ref<Texture> pAlbedoTexture,
                             ref<Texture> pColorTexture, ref<Texture> pEmissionTexture,
                             ref<Texture> pMotionVectorTexture,
                             ref<Texture> pPositionNormalFwidthTexture,
                             ref<Texture> pPrevLinearZAndNormalTexture,
                             ref<Texture> pDebugTexture
        );

    void computeDerivVerification(RenderContext* pRenderContext, const SVGFRenderData& renderData);


    bool mBuffersNeedClear = false;

    // SVGF parameters
    bool    mFilterEnabled       = true;
    int32_t mFilterIterations    =  4;
    int32_t mFeedbackTap         = -1;
    float   mVarainceEpsilon     =  1e-4f;
    int mDerivativeIteration     =  0;

    ref<Buffer> mReadbackBuffer[3];

    ref<FullScreenPass> mpDerivativeVerify;
    ref<Fbo> mpDerivativeVerifyFbo;
    float mDelta;
    ref<Texture> mpFuncOutputLower;
    ref<Texture> mpFuncOutputUpper;

    // Intermediate framebuffers
    ref<Fbo> mpPingPongFbo[2];
    ref<Fbo> mpLinearZAndNormalFbo;
    ref<Fbo> mpFilteredPastFbo;
    ref<Fbo> mpCurReprojFbo;
    ref<Fbo> mpPrevReprojFbo;
    ref<Fbo> mpFilteredIlluminationFbo;
    ref<Fbo> mpFinalFbo;

    ref<FullScreenPass> compactingPass;
    ref<Buffer> pdaRawOutputBuffer[3];
    ref<Buffer> pdaCompactedBuffer[3];
    void runCompactingPass(RenderContext* pRenderContext, int idx, int n);
    void clearRawOutputBuffer(RenderContext* pRenderContext, int idx);

    ref<ComputePass> summingPass;
    ref<FullScreenPass> bufferToTexturePass;
    ref<Fbo> bufferToTextureFbo;
    ref<Buffer> pdaPingPongSumBuffer[2];
    void reduceParameter(RenderContext* pRenderContext, SVGFParameter<float4>& param, int offset);

    std::unique_ptr<ParallelReduction> mpParallelReduction; 

    int mDatasetIndex = 0;
    SVGFTrainingDataset mTrainingDataset;

    struct
    {
        int2 minP;
        int2 maxP;
    } mPatch;
    bool mPatchingEnabled;

    void setPatchingState(ref<FullScreenPass> fs);

    struct ParameterMetaInfo
    {
        // float4 is max allowed size
        SVGFParameter<float4>* mAddress;
        float4 momentum;
        float4 ssgrad;
        int mNumElements;
        std::string mName;
    };
    std::vector<ParameterMetaInfo> mParameterReflector;

    //  manually registers parameter (but it still is auto trained)
    void registerParameterManual(SVGFParameter<float4>* param, int cnt, const std::string& name);

    // registers parameter into list of parameters so we automatically train it
    template<typename T>
    void registerParameterUM(SVGFParameter<T>& param, const std::string& name)
    {
        registerParameterManual((SVGFParameter<float4>*)&param, sizeof(T) / sizeof(float), name);
    }

    // we want to optimize parameters per pass to get a little bit of extra tuning
    // da is short for derivative accum

    struct {
        ref<Texture> pLinearZAndNormal;
        ref<FullScreenPass> sPass;
    } mPackLinearZAndNormalState;

    struct {
        ref<Texture> ptIllum;
        ref<Texture> ptHistoryLen;
        ref<Texture> ptMoments;

        ref<Texture> pPrevFiltered;
        ref<Texture> pPrevMoments;
        ref<Texture> pPrevHistoryLength;

        SVGFParameter<float> mAlpha;
        SVGFParameter<float> mMomentsAlpha;

        SVGFParameter<float3> mLuminanceParams;

        SVGFParameter<float[4]> mParams;
        SVGFParameter<float[3]> mKernel;

        ref<FullScreenPass> sPass;
        ref<FullScreenPass> dPass;
    } mReprojectState;

    struct {
        ref<Buffer> pdaHistoryLen;

        ref<Texture> pCurIllum;
        ref<Texture> pCurMoments;
        ref<Texture> pCurHistoryLength;

        SVGFParameter<float3> mSigma;

        SVGFParameter<float3> mLuminanceParams;
        SVGFParameter<float[3]> mWeightFunctionParams;

        SVGFParameter<float> mVarianceBoostFactor;

        ref<FullScreenPass> sPass;
        ref<FullScreenPass> dPass;
    } mFilterMomentsState;

    struct {

        struct PerIterationState
        {
            ref<Texture> pgIllumination; // saved illumination for this iteration

            SVGFParameter<float3> mSigma;

            SVGFParameter<float[3]> mWeightFunctionParams;
            SVGFParameter<float3> mLuminanceParams;

            SVGFParameter<float[2][2]> mVarianceKernel;
            SVGFParameter<float[3]> mKernel;
        };

        ref<Texture> mSaveIllum;

        std::vector<PerIterationState> mIterationState;

        ref<FullScreenPass> sPass;
        ref<FullScreenPass> dPass;
    } mAtrousState;

    struct {
        ref<Buffer> pdaIllumination;

        ref<Texture> pFinalFiltered;

        ref<FullScreenPass> sPass;
        ref<FullScreenPass> dPass;
    } mFinalModulateState;

    struct
    {
        ref<Fbo> pGaussianFbo[2];

        ref<Texture> pGaussianXInput;
        ref<Texture> pGaussianYInput;

        ref<Texture> pFilteredGaussian;
        ref<Texture> pReferenceGaussian;

        ref<FullScreenPass> sGaussianPass;
        ref<FullScreenPass> dGaussianPass;
        ref<FullScreenPass> dPass;
    } mLossState;
};

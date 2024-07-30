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
#include "SVGFAtrous.h"

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
    ref<FilterParameterReflector> mpParameterReflector;

    ref<Scene> pScene;

    SVGFRenderData mRenderData;

    bool mTrained = false;
    int mEpoch = 0;
    void runNextTrainingTask(RenderContext* pRenderContext);
    void clearTrainingBuffers(RenderContext* pRenderContext);
    void reduceAllData(RenderContext* pRenderContext);
    float getAverageGradient(float* ptr, int baseOffset, int sampledFrames);
    float calculateBaseAdjustment(float gradient, float& momentum, float& ssgrad);
    void updateParameters(RenderContext* pRenderContext, int sampledFrames);
    void printLoss(RenderContext* pRenderContext, int sampledFrames);

    void runDerivativeTest(RenderContext* pRenderContext, const RenderData& renderData);
    void runTrainingAndTesting(RenderContext* pRenderContext, const RenderData& renderData);


    void allocateFbos(uint2 dim, RenderContext* pRenderContext);
    void clearBuffers(RenderContext* pRenderContext, const SVGFRenderData& renderData);

    void runSvgfFilter(RenderContext* pRenderContext, SVGFRenderData& renderData, bool updateInternalBuffers);
    void computeLinearZAndNormal(RenderContext* pRenderContext, SVGFRenderData& renderData);
    void computeReprojection(RenderContext* pRenderContext, SVGFRenderData& renderData);
    void computeFilteredMoments(RenderContext* pRenderContext, SVGFRenderData& svgfrd);
    void computeAtrousDecomposition(RenderContext* pRenderContext, SVGFRenderData& renderData, bool updateInternalBuffers);
    void computeGaussian(RenderContext* pRenderContext, ref<Texture> tex, ref<Texture> storageLocation, bool saveTextures);

    void computeDerivatives(RenderContext* pRenderContext, SVGFRenderData& renderData, bool useLoss);
    void computeLoss(RenderContext* pRenderContext, SVGFRenderData& renderData, bool saveInternalState);
    void computeDerivGaussian(RenderContext* pRenderContext);
    void computeDerivFinalModulate(RenderContext* pRenderContext, SVGFRenderData& renderData);
    void computeDerivAtrousDecomposition(RenderContext* pRenderContext, SVGFRenderData& renderData);
    void computeDerivFilteredMoments(RenderContext* pRenderContext, SVGFRenderData& renderData);
    void computeDerivReprojection(RenderContext* pRenderContext, SVGFRenderData& renderData);

    void computeDerivVerification(RenderContext* pRenderContext, const SVGFRenderData& renderData);


    void saveLossBuffers(RenderContext* pRenderContext, SVGFRenderData& renderData);
    void updateLossBuffers(RenderContext* pRenderContext, SVGFRenderData& renderData);

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



    ref<ComputePass> summingPass;
    ref<FullScreenPass> bufferToTexturePass;
    ref<Fbo> bufferToTextureFbo;
    ref<Buffer> pdaPingPongSumBuffer[2];
    int reduceParameter(RenderContext* pRenderContext, ParameterMetaInfo& param, int offset);

    std::unique_ptr<ParallelReduction> mpParallelReduction; 

    int mDatasetIndex = 0;
    int mBatchSize = 0;
    bool mBackPropagatingState = false;
    bool mFirstBackPropIteration = true;
    int mReductionAddress = 0;
    float mBetaMomentumCorrection;
    float mBetaSsgradCorrection;
    SVGFTrainingDataset mTrainingDataset;

    // we want to optimize parameters per pass to get a little bit of extra tuning
    // da is short for derivative accum
    struct {
        ref<FullScreenPass> sPass;
    } mPackLinearZAndNormalState;

    struct {
        ref<Buffer> pdaIllumination;
        ref<Buffer> pdaMoments;
        ref<Buffer> pdaHistoryLength;

        SVGFParameter<float> mAlpha;
        SVGFParameter<float> mMomentsAlpha;

        SVGFParameter<float3> mLuminanceParams;

        SVGFParameter<float[4]> mParams;
        SVGFParameter<float[3]> mKernel;

        SVGFParameter<float[kNumReprojectionMlpWeights]> mTemporalMlpWeights;

        ref<FullScreenPass> sPass;
        ref<FullScreenPass> dPass;
    } mReprojectState;

    struct {
        ref<Buffer> pdaHistoryLen;

        SVGFParameter<float3> mSigma;

        SVGFParameter<float3> mLuminanceParams;
        SVGFParameter<float[3]> mWeightFunctionParams;

        SVGFParameter<float> mVarianceBoostFactor;

        ref<FullScreenPass> sPass;
        ref<FullScreenPass> dPass;
    } mFilterMomentsState;

    ref<SVGFAtrousSubpass> mpAtrousSubpass;

    struct {
        ref<Buffer> pdaIllumination;

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

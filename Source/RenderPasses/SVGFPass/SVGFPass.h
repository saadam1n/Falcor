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

using namespace Falcor;

struct SVGFRenderData
{
    ref<Texture> pAlbedoTexture;
    ref<Texture> pColorTexture;
    ref<Texture> pEmissionTexture;
    ref<Texture> pWorldPositionTexture;
    ref<Texture> pWorldNormalTexture;
    ref<Texture> pPosNormalFwidthTexture;
    ref<Texture> pLinearZTexture;
    ref<Texture> pMotionVectorTexture;
    ref<Texture> pPrevLinearZAndNormalTexture;
    ref<Texture> pOutputTexture;
    ref<Texture> pDebugTexture;
    ref<Texture> pDerivVerifyTexture;

    SVGFRenderData(const RenderData& renderData);
};

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
    ref<Scene> pScene;

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
    void computeAtrousDecomposition(RenderContext* pRenderContext, ref<Texture> pAlbedoTexture, bool nonFiniteDiffPass);

    void runSvgfFilter(RenderContext* pRenderContext, const SVGFRenderData& renderData, bool shouldCollectDerivatives);
    void computeDerivatives(RenderContext* pRenderContext, const SVGFRenderData& renderData);
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

    ref<Buffer> createAccumulationBuffer(ref<Device> pDevice, int bytes_per_elem = sizeof(int4), bool need_readback = false);
    ref<Texture> createFullscreenTexture(ref<Device> pDevice, ResourceFormat fmt = ResourceFormat::RGBA32Float);

    bool mBuffersNeedClear = false;

    // SVGF parameters
    bool    mFilterEnabled       = true;
    int32_t mFilterIterations    = 4;
    int32_t mFeedbackTap         = 1;
    float   mVarainceEpsilon     = 1e-4f;
    int mDerivativeIteration;

    // SVGF passes
    ref<FullScreenPass> mpPackLinearZAndNormal;
    ref<FullScreenPass> mpReprojection;
    ref<FullScreenPass> mpFilterMoments;
    ref<FullScreenPass> mpAtrous;
    ref<FullScreenPass> mpFinalModulate;

    ref<Buffer> mReadbackBuffer;

    ref<FullScreenPass> mpDerivativeVerify;
    ref<Fbo> mpDerivativeVerifyFbo;
    float mDelta;
    ref<Texture> mpFuncOutputLower;
    ref<Texture> mpFuncOutputUpper;

    ref<Fbo> mpDummyFullscreenFbo;

    // Intermediate framebuffers
    ref<Fbo> mpPingPongFbo[2];
    ref<Fbo> mpLinearZAndNormalFbo;
    ref<Fbo> mpFilteredPastFbo;
    ref<Fbo> mpCurReprojFbo;
    ref<Fbo> mpPrevReprojFbo;
    ref<Fbo> mpFilteredIlluminationFbo;
    ref<Fbo> mpFinalFbo;

    ref<FullScreenPass> compactingPass;
    ref<Buffer> pdaRawOutputBuffer[2];
    ref<Buffer> pdaCompactedBuffer[2];
    void runCompactingPass(RenderContext* pRenderContext, int idx, int n);

    ref<ComputePass> summingPass;
    ref<Buffer> pdaPingPongSumBuffer[2];

    // we want to optimize parameters per pass to get a little bit of extra tuning
    // da is short for derivative accum

    struct {

    } mPackLinearZAndNormalState;

    struct {
        ref<Texture> ptIllum;
        ref<Texture> ptHistoryLen;
        ref<Texture> ptMoments;

        ref<Texture> pPrevIllum;

        ref<Buffer> pdaLuminanceParams;
        ref<Buffer> pdaReprojKernel;
        ref<Buffer> pdaReprojParams;
        ref<Buffer> pdaAlpha;
        ref<Buffer> pdaMomentsAlpha;

        float dvAlpha;
        float dvMomentsAlpha;

        float3 dvLuminanceParams;

        float dvParams[4];
        float dvKernel[3];

        ref<FullScreenPass> dPass;
    } mReprojectState;

    struct {
        ref<Buffer> pdaHistoryLen;

        ref<Texture> pLumVarTex;

        ref<Buffer> pdaVarianceBoostFactor;
        ref<Buffer> pdaLuminanceParams;
        ref<Buffer> pdaWeightFunctionParams;

        ref<Buffer> pdaSigma;

        float   dvSigmaL;
        float   dvSigmaZ;
        float   dvSigmaN;

        float3 dvLuminanceParams;
        float dvWeightFunctionParams[3];

        float dvVarianceBoostFactor;

        ref<FullScreenPass> dPass;
    } mFilterMomentsState;

    struct {

        struct PerIterationState
        {
            ref<Texture> pgIllumination; // saved illumination for this iteration

            ref<Buffer> pdaKernel;
            ref<Buffer> pdaVarianceKernel;
            ref<Buffer> pdaLuminanceParams;
            ref<Buffer> pdaWeightFunctionParams;
            ref<Buffer> pdaSigma;

            float   dvSigmaL;
            float   dvSigmaZ;
            float   dvSigmaN;

            float dvWeightFunctionParams[3];
            float3 dvLuminanceParams;

            float dvVarianceKernel[2][2];
            float dvKernel[3];
        };

        ref<Texture> mSaveIllum;

        std::vector<PerIterationState> mIterationState;

        ref<FullScreenPass> dPass;
    } mAtrousState;




    struct {
        ref<Buffer> pdaIllumination;
        ref<Buffer> pdrFilteredImage;

        ref<FullScreenPass> dPass;
    } mFinalModulateState;
};

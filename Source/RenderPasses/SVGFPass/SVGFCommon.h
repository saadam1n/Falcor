// ooga booga copyright notice
#pragma once

#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Core/Pass/FullScreenPass.h"

using namespace Falcor;

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

    const char kDummyFullScreenShader[]       = "RenderPasses/SVGFPass/SVGFDummyFullScreenPass.ps.slang";

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
    const char kInternalBufferPreviousFiltered[] = "Previous Filtered";
    const char kInternalBufferPreviousReference[] = "Previous Reference";

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
    const char kOutputCenterLoss[] = "CenterLoss";
    const char kOutputGradientLoss[] = "GradientLoss";
    const char kOutputTemporalLoss[] = "TemporalLoss";

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

    const float   dvSigmaL              = 1.0f;
    const float   dvSigmaZ              = 1.0;
    const float   dvSigmaN              = 1.0f;
    const float   dvAlpha               = 0.05f;
    const float   dvMomentsAlpha        = 0.2f;

    // x is L
    // y is Z
    // z is N
    const float3  dvSigma = float3(dvSigmaL, dvSigmaZ, dvSigmaN);

    const float dvWeightFunctionParams[3] {1.0, 1.0, 1.0};
}

template<typename T>
struct SVGFParameter
{
    ref<Buffer> da;
    T dv;

    void clearBuffer(RenderContext* pRenderContext)
    {
        pRenderContext->clearUAV(da->getUAV().get(), uint4(0));
    }
};

class SVGFUtilitySet : public Object
{
public:
    SVGFUtilitySet(ref<Device> pDevice, int minX, int minY, int maxX, int maxY);
    void allocateFbos(uint2 dim, RenderContext* pRenderContext);

    ref<Buffer> createAccumulationBuffer(int bytes_per_elem = sizeof(float4), bool need_reaback = false);
    ref<Texture> createFullscreenTexture(ResourceFormat fmt = ResourceFormat::RGBA32Float);
    ref<FullScreenPass> createFullscreenPassAndDumpIR(const std::string& path);

    ref<Fbo> getDummyFullscreenFbo();
    void executeDummyFullscreenPass(RenderContext* pRenderContext, ref<Texture> tex);

    void runCompactingPass(RenderContext* pRenderContext, int idx, int n);
    void clearRawOutputBuffer(RenderContext* pRenderContext, int idx);

    ref<Buffer> mpdaRawOutputBuffer[2];
    ref<Buffer> mpdrCompactedBuffer[2];

    void setPatchingState(ref<FullScreenPass> fsPass);
private:
    ref<Device> mpDevice;

    int mBufferMemUsage = 0;
    int mTextureMemUsage = 0;

    ref<Fbo> mpDummyFullscreenFbo;
    ref<FullScreenPass> mpDummyFullscreenPass;

    ref<FullScreenPass> mpCompactingPass;

public:
    int2 mPatchMinP;
    int2 mPatchMaxP;
};

struct ParameterMetaInfo
{
    // float4 is max allowed size
    float* mAddress;
    ref<Buffer> mAccum;
    int mNumElements;
    std::string mName;

    // parameters to use during learning
    std::vector<float> momentum;
    std::vector<float> ssgrad;
};

class FilterParameterReflector : public Object
{
public:
    FilterParameterReflector(ref<SVGFUtilitySet> pUtilities);

    //  manually registers parameter (but it still is auto trained)
    void registerParameterManual(float* addr, ref<Buffer>* accum, int cnt, const std::string& name);

    // registers parameter into list of parameters so we automatically train it
    template<typename T>
    void registerParameterAuto(SVGFParameter<T>& param, const std::string& name)
    {
        registerParameterManual((float*) &param.dv, &param.da, sizeof(T) / sizeof(float), name);
    }

    size_t getNumParams();
    int getPackedStride();

    std::vector<ParameterMetaInfo> mRegistry;
private:
    ref<SVGFUtilitySet> mpUtilities;
};
#define REGISTER_PARAMETER(reflector, x) reflector->registerParameterAuto(x, #x);

// the renderdata class contains external inputs and outputs for the SVGF algorithm
struct SVGFRenderData
{
public:
    SVGFRenderData(ref<Device> pDevice, ref<SVGFUtilitySet> utilities);
    SVGFRenderData(ref<Device> pDevice, ref<SVGFUtilitySet> utilities, const RenderData& renderData);

    void copyTextureReferences(const RenderData& renderData);

    ref<Texture> pAlbedoTexture;
    ref<Texture> pColorTexture;
    ref<Texture> pEmissionTexture;
    ref<Texture> pWorldPositionTexture;
    ref<Texture> pWorldNormalTexture;
    ref<Texture> pPosNormalFwidthTexture;
    ref<Texture> pLinearZTexture;
    ref<Texture> pMotionVectorTexture;
    ref<Texture> pPrevLinearZAndNormalTexture; // todo: move this out of render data because it is an internal buffer
    ref<Texture> pOutputTexture;
    ref<Texture> pDebugTexture;
    ref<Texture> pDerivVerifyTexture;

    // only used in training
    ref<Texture> pReferenceTexture;
    ref<Texture> pLossTexture;
    ref<Texture> pCenterLossTexture;
    ref<Texture> pGradientLossTexture;
    ref<Texture> pTemporalLossTexture;
    ref<Texture> pPrevFiltered;
    ref<Texture> pPrevReference;

    // access the texture table
    ref<Texture>& fetchTexTable(const std::string& s);
    ref<Buffer>& fetchBufTable(const std::string& s);

    void saveInternalTex(RenderContext* pRenderContext, const std::string& s, ref<Texture> tex, bool shouldSaveRevisions);
    ref<Texture> fetchInternalTex(const std::string& s);
    void pushInternalBuffers(RenderContext* pRenderContext);
    void popInternalBuffers(RenderContext* pRenderContext);
protected:
    // keep track of this for whatever reason
    ref<Device> mpDevice;
    // utils
    ref<SVGFUtilitySet> mpUtilities;
private:
    // the advantage of using a ref here is that we do not have to blit
    std::map<std::string, ref<Texture>> mTextureTable;
    std::map<std::string, ref<Buffer>> mBufferTable;

    struct InternalTexture
    {
        ref<Texture> mSavedTexture;
        bool mShouldSaveRevisions;
        // all saved revisions of this texture
        std::vector<Bitmap::UniqueConstPtr> mSavedRevisions;

        InternalTexture operator=(const InternalTexture& other) = delete;
    };

    std::map<std::string, InternalTexture> mInternalTextureMappings;
    int mInternalRegistryFrameCount;
    std::vector<std::future<void>> mAsyncReadOperations;
};

struct SVGFTrainingDataset : public SVGFRenderData
{
public:
    SVGFTrainingDataset(ref<Device> pDevice, ref<SVGFUtilitySet> utilities, const std::string& folder);
    bool loadNext(RenderContext* pRenderContext);
    bool loadPrev(RenderContext* pRenderContext);
    void reset();

    // preload all bitmaps, if not already
    void preloadBitmaps();
private:
    // the folder containing the dataset
    std::string mFolder;
    // whatever sample we are reading from
    int mDatasetIndex;
    // cache of preloaded bitmaps
    std::map<std::string, Bitmap::UniqueConstPtr> mCachedBitmaps;
    // cache of texture name to pointer mappings
    std::map<std::string, ref<Texture>> mTextureNameMappings;

    // list of bitmaps that are being currently preloaded
    std::map<std::string, std::future<Bitmap::UniqueConstPtr>> mPreloadingBitmaps;
    // whether a preload request was submitted in the past
    bool mPreloaded = false;

    bool atValidIndex() const;
    bool loadCurrent(RenderContext* pRenderContext);
    std::string getSampleBufferPath(const std::string& buffer) const;
    static Bitmap::UniqueConstPtr readBitmapFromFile(const std::string& path);
    void loadSampleBuffer(RenderContext* pRenderContext, ref<Texture> tex, const std::string& buffer);
};



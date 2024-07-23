#include "SVGFCommon.h"
#include <fstream>

using namespace Falcor;



SVGFUtilitySet::SVGFUtilitySet(ref<Device> pDevice, int minX, int minY, int maxX, int maxY) : mpDevice(pDevice), mPatchMinP(minX, minY), mPatchMaxP(maxX, maxY)
{
    // set some general utility states
    mpCompactingPass = createFullscreenPassAndDumpIR(kBufferShaderCompacting);

    mpdaRawOutputBuffer[0] = createAccumulationBuffer(sizeof(float4) * 50);
    mpdaRawOutputBuffer[1] = createAccumulationBuffer(sizeof(float4) * 49);
    for (int i = 0; i < 2; i++)
    {
        mpdrCompactedBuffer[i] = createAccumulationBuffer();
    }
}

void SVGFUtilitySet::allocateFbos(uint2 dim, RenderContext* pRenderContext)
{
    {
        // contains a debug buffer for whatever we want to store
        Fbo::Desc desc;
        desc.setSampleCount(0);
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float);
        desc.setColorTarget(1, Falcor::ResourceFormat::RGBA32Float);
        desc.setColorTarget(2, Falcor::ResourceFormat::RGBA32Float);
        desc.setColorTarget(3, Falcor::ResourceFormat::RGBA32Float);
        mpDummyFullscreenFbo = Fbo::create2D(mpDevice, dim.x, dim.y, desc);
    }
}

ref<Buffer> SVGFUtilitySet::createAccumulationBuffer(int bytes_per_elem, bool need_reaback) {
    int2 patchDim = mPatchMaxP - mPatchMinP;
    int patchNumPixels = patchDim.x * patchDim.y;

    mBufferMemUsage += patchNumPixels * bytes_per_elem;
    return make_ref<Buffer>(mpDevice, bytes_per_elem * patchNumPixels, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, need_reaback ? MemoryType::ReadBack : MemoryType::DeviceLocal, nullptr);
}

ref<Texture> SVGFUtilitySet::createFullscreenTexture(ResourceFormat fmt)
{
    mTextureMemUsage += numPixels * sizeof(float4); // TODO: take format into consideration
    return make_ref<Texture>(mpDevice, Resource::Type::Texture2D, fmt, screenWidth, screenHeight,  1, 1, 1, 1, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, nullptr);
}

ref<FullScreenPass> SVGFUtilitySet::createFullscreenPassAndDumpIR(const std::string& path)
{
    ProgramDesc desc;
    desc.compilerFlags |= SlangCompilerFlags::DumpIntermediates;
    desc.addShaderLibrary(path).psEntry("main");
    return FullScreenPass::create(mpDevice, desc);
}

ref<Fbo> SVGFUtilitySet::getDummyFullscreenFbo()
{
    return mpDummyFullscreenFbo;
}

void SVGFUtilitySet::runCompactingPass(RenderContext* pRenderContext, int idx, int n)
{
    FALCOR_PROFILE(pRenderContext, "Compacting " + std::to_string(idx));

    setPatchingState(mpCompactingPass);

    auto compactingCB = mpCompactingPass->getRootVar()["CompactingCB"];
    compactingCB["drIllumination"] = mpdaRawOutputBuffer[idx];
    compactingCB["daIllumination"] = mpdrCompactedBuffer[idx];
    compactingCB["gAlbedo"] = getDummyFullscreenFbo()->getColorTexture(0);

    compactingCB["elements"] = n;
    // compact the raw output
    mpCompactingPass->execute(pRenderContext, getDummyFullscreenFbo());
}

void SVGFUtilitySet::clearRawOutputBuffer(RenderContext* pRenderContext, int idx)
{
    FALCOR_PROFILE(pRenderContext, "Clr Raw Out " + std::to_string(idx));
    pRenderContext->clearUAV(mpdaRawOutputBuffer[idx]->getUAV().get(), uint4(0));
}

void SVGFUtilitySet::setPatchingState(ref<FullScreenPass> fsPass)
{
    auto patchInfo = fsPass->getRootVar()["PatchInfo"];

    patchInfo["patchMinP"] = mPatchMinP;
    patchInfo["patchMaxP"] = mPatchMaxP;
}

FilterParameterReflector::FilterParameterReflector(ref<SVGFUtilitySet> pUtilities) : mpUtilities(pUtilities) {}

void FilterParameterReflector::registerParameterManual(float* addr, ref<Buffer>* accum, int cnt, const std::string& name)
{
    *accum = mpUtilities->createAccumulationBuffer(sizeof(float4) * ((cnt + 3) / 4));

    ParameterMetaInfo pmi;

    pmi.mAddress = addr;
    pmi.mAccum = *accum;
    pmi.mNumElements = cnt;
    pmi.mName = name;

    pmi.momentum.resize(cnt);
    pmi.ssgrad.resize(cnt);

    mRegistry.push_back(pmi);
}

size_t FilterParameterReflector::getNumParams()
{
    return mRegistry.size();
}

SVGFRenderData::SVGFRenderData(ref<Device> pDevice, ref<SVGFUtilitySet> utilities) : mpDevice(pDevice), mpUtilities(utilities), mInternalRegistryFrameCount(0) {}

SVGFRenderData::SVGFRenderData(ref<Device> pDevice, ref<SVGFUtilitySet> utilities, const RenderData& renderData) : SVGFRenderData(pDevice, utilities)
{
    copyTextureReferences(renderData);
}

void SVGFRenderData::copyTextureReferences(const RenderData& renderData)
{
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

    // loss buffers
    pLossTexture = renderData.getTexture(kOutputLoss);
    pCenterLossTexture = renderData.getTexture(kOutputCenterLoss);
    pGradientLossTexture = renderData.getTexture(kOutputGradientLoss);
    pTemporalLossTexture = renderData.getTexture(kOutputTemporalLoss);
    pReferenceTexture = renderData.getTexture(kOutputReference);
    pPrevFiltered = renderData.getTexture(kInternalBufferPreviousFiltered);
    pPrevReference = renderData.getTexture(kInternalBufferPreviousReference);
}

ref<Texture>& SVGFRenderData::fetchTexTable(const std::string& s)
{
    return mTextureTable[s];
}

ref<Buffer>& SVGFRenderData::fetchBufTable(const std::string& s)
{
    return mBufferTable[s];
}

void SVGFRenderData::saveInternalTex(RenderContext* pRenderContext, const std::string& s, ref<Texture> tex)
{
    if (mInternalTextureMappings.count(s) == 0)
    {
        mInternalTextureMappings[s].mSavedTexture = mpUtilities->createFullscreenTexture();
    }

    pRenderContext->blit(tex->getSRV(), mInternalTextureMappings[s].mSavedTexture->getRTV());
}

ref<Texture> SVGFRenderData::fetchInternalTex(const std::string& s)
{
    return mInternalTextureMappings[s].mSavedTexture;
}

void SVGFRenderData::pushInternalBuffers(RenderContext* pRenderContext)
{
    int saveIndex = mInternalRegistryFrameCount++;
    int requiredSize = mInternalRegistryFrameCount;

    for (auto& [s, internalTex] : mInternalTextureMappings)
    {
        CopyContext::ReadTextureTask::SharedPtr ptr = pRenderContext->asyncReadTextureSubresource(internalTex.mSavedTexture.get(), 0);

        // save this current texture
        // allocate more slots if not avaiable yet
        if (internalTex.mSavedRevisions.size() < requiredSize)
        {
            internalTex.mSavedRevisions.resize(requiredSize);
        }

        // allocate bitmap if not allocated already
        if (!internalTex.mSavedRevisions[saveIndex].get())
        {
            internalTex.mSavedRevisions[saveIndex] = std::move(Bitmap::create(screenWidth, screenHeight, ResourceFormat::RGBA32Float, nullptr));
        }

        auto async_download = [](CopyContext::ReadTextureTask::SharedPtr src, uint8_t* dst)
        {
            src->getData((void*)dst, numPixels * sizeof(float4));
        };

        //mAsyncReadOperations.push_back(std::async(std::launch::async, async_download, ptr, internalTex.mSavedRevisions[saveIndex]->getData()));
        async_download(ptr, internalTex.mSavedRevisions[saveIndex]->getData());
    }
}

void SVGFRenderData::popInternalBuffers(RenderContext* pRenderContext)
{
    while (!mAsyncReadOperations.empty())
    {
        mAsyncReadOperations.back().get();
        mAsyncReadOperations.pop_back();
    }

    int readIndex = --mInternalRegistryFrameCount;

    for (auto& [s, internalTex] : mInternalTextureMappings)
    {
        pRenderContext->updateTextureData(internalTex.mSavedTexture.get(), (const void*)internalTex.mSavedRevisions[readIndex]->getData());
    }
}

SVGFTrainingDataset::SVGFTrainingDataset(ref<Device> pDevice, ref<SVGFUtilitySet> utilities, const std::string& folder) : SVGFRenderData(pDevice, utilities), mFolder(folder), mSampleIdx(0) {
#define MarkDatasetTexture(x) p##x##Texture = mpUtilities->createFullscreenTexture(); mTextureNameMappings[kDataset##x] = p##x##Texture;

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

    pLossTexture = mpUtilities->createFullscreenTexture();
    pCenterLossTexture = mpUtilities->createFullscreenTexture();
    pGradientLossTexture = mpUtilities->createFullscreenTexture();
    pTemporalLossTexture = mpUtilities->createFullscreenTexture();
    pPrevLinearZAndNormalTexture = mpUtilities->createFullscreenTexture();
    pOutputTexture = mpUtilities->createFullscreenTexture();
    pDebugTexture = mpUtilities->createFullscreenTexture();
    pDerivVerifyTexture = mpUtilities->createFullscreenTexture();

    pPrevFiltered = mpUtilities->createFullscreenTexture();
    pPrevReference = mpUtilities->createFullscreenTexture();
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

void SVGFTrainingDataset::preloadBitmaps()
{
    if(!mPreloaded)
    {
        mPreloaded = true;

        // preload all textures
        std::cout << "Starting async dataset preload... this may cause high background CPU usuage for a bit!" << std::endl;

        std::map<std::string, std::future<Bitmap::UniqueConstPtr>> preloadTasks;
        while(atValidIndex())
        {
            for(auto [buffer, tex] : mTextureNameMappings)
            {
                std::string path = getSampleBufferPath(buffer);
                mPreloadingBitmaps[path] = std::async(std::launch::async, &readBitmapFromFile, path);
            }

            mSampleIdx++;
        }

        mSampleIdx = 0;
    }
}

bool SVGFTrainingDataset::atValidIndex() const
{
    return std::filesystem::exists(getSampleBufferPath(kDatasetColor));
}

std::string SVGFTrainingDataset::getSampleBufferPath(const std::string& buffer) const
{
    return mFolder + std::to_string(mSampleIdx) + "-" + buffer + ".exr";
}

Bitmap::UniqueConstPtr SVGFTrainingDataset::readBitmapFromFile(const std::string& path)
{
    std::string name = std::filesystem::path(path).filename().string();
    std::string cachePath = "C:\\FalcorFiles\\Cache\\" + std::to_string(getFileModifiedTime(path)) + "-" + name + ".bin";

    Bitmap::UniqueConstPtr bitmap;
    if(!std::filesystem::exists(cachePath))
    {
        bitmap = std::move(Bitmap::createFromFile(path, true));

        std::ofstream fs(cachePath, std::ios::binary);
        fs.write((char*)bitmap->getData(), numPixels * sizeof(float4));
        fs.close();
    }
    else
    {
        auto len = std::filesystem::file_size(cachePath);

        bitmap = std::move(Bitmap::create(screenWidth, screenHeight, ResourceFormat::RGBA32Float, nullptr));

        // do this all with DMA instead of CPU memcpys
        std::ifstream fs(cachePath, std::ios::binary);
        fs.read((char*)bitmap->getData(), len);
        fs.close();
    }

    return bitmap;
}

void SVGFTrainingDataset::loadSampleBuffer(RenderContext* pRenderContext, ref<Texture> tex, const std::string& buffer)
{
    std::string path = getSampleBufferPath(buffer);

    if(mCachedBitmaps.count(path) == 0)
    {
        // check if already loaded
        if (mPreloadingBitmaps.count(path) == 1)
        {
            mCachedBitmaps[path] = std::move(mPreloadingBitmaps[path].get());
            mPreloadingBitmaps.erase(path);
        }
        else
        {
            // manually load
            mCachedBitmaps[path] = std::move(readBitmapFromFile(path));
        }
    }

    if(pRenderContext) {
        pRenderContext->updateTextureData(tex.get(), (const void*)mCachedBitmaps[path]->getData());
    }
}

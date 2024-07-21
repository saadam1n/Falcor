#include "SVGFCommon.h"
#include <fstream>

using namespace Falcor;

SVGFUtilitySet::SVGFUtilitySet(ref<Device> pDevice) : mpDevice(pDevice)
{

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
    mBufferMemUsage += numPixels * bytes_per_elem;
    return make_ref<Buffer>(mpDevice, bytes_per_elem * numPixels, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, need_reaback ? MemoryType::ReadBack : MemoryType::DeviceLocal, nullptr);
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

    // loss buffers
    pLossTexture = renderData.getTexture(kOutputLoss);
    pCenterLossTexture = renderData.getTexture(kOutputCenterLoss);
    pGradientLossTexture = renderData.getTexture(kOutputGradientLoss);
    pTemporalLossTexture = renderData.getTexture(kOutputTemporalLoss);
    pReferenceTexture = renderData.getTexture(kOutputReference);
    pPrevFiltered = renderData.getTexture(kInternalBufferPreviousFiltered);
    pPrevReference = renderData.getTexture(kInternalBufferPreviousReference);
}

SVGFTrainingDataset::SVGFTrainingDataset(ref<Device> pDevice, ref<SVGFUtilitySet> utilities, const std::string& folder) : mpUtilities(utilities), mpDevice(pDevice), mFolder(folder), mSampleIdx(0) {
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

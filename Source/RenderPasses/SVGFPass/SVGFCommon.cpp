#include "SVGFCommon.h"
#include <fstream>

using namespace Falcor;

namespace SVGFUtil
{
    ref<Buffer> createAccumulationBuffer(ref<Device> pDevice, int bytes_per_elem, bool need_reaback) {
        return make_ref<Buffer>(pDevice, bytes_per_elem * numPixels, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, need_reaback ? MemoryType::ReadBack : MemoryType::DeviceLocal, nullptr);
    }

    ref<Texture> SVGFUtil::createFullscreenTexture(ref<Device> pDevice, ResourceFormat fmt)
    {
        return make_ref<Texture>(pDevice, Resource::Type::Texture2D, fmt, screenWidth, screenHeight,  1, 1, 1, 1, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, nullptr);
    }

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

SVGFTrainingDataset::SVGFTrainingDataset(ref<Device> pDevice, const std::string& folder) : mFolder(folder), mSampleIdx(0) {
#define MarkDatasetTexture(x) p##x##Texture = SVGFUtil::createFullscreenTexture(pDevice); mTextureNameMappings[kDataset##x] = p##x##Texture;

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

    pLossTexture = SVGFUtil::createFullscreenTexture(pDevice);
    pCenterLossTexture = SVGFUtil::createFullscreenTexture(pDevice);
    pGradientLossTexture = SVGFUtil::createFullscreenTexture(pDevice);
    pTemporalLossTexture = SVGFUtil::createFullscreenTexture(pDevice);
    pPrevLinearZAndNormalTexture = SVGFUtil::createFullscreenTexture(pDevice);
    pOutputTexture = SVGFUtil::createFullscreenTexture(pDevice);
    pDebugTexture = SVGFUtil::createFullscreenTexture(pDevice);
    pDerivVerifyTexture = SVGFUtil::createFullscreenTexture(pDevice);

    pPrevFiltered = SVGFUtil::createFullscreenTexture(pDevice);
    pPrevReference = SVGFUtil::createFullscreenTexture(pDevice);
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
        std::cout << "Preloading the dataset..." << std::endl;

        std::map<std::string, std::future<Bitmap::UniqueConstPtr>> preloadTasks;
        while(atValidIndex())
        {
            for(auto [buffer, tex] : mTextureNameMappings)
            {
                std::string path = getSampleBufferPath(buffer);
                preloadTasks[path] = std::async(std::launch::async, &readBitmapFromFile, path);
            }

            mSampleIdx++;
        }

        mSampleIdx = 0;

        for(auto& [path, bitmap] : preloadTasks)
        {
            mPreloadedBitmaps[path] = std::move(bitmap.get());
        }
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

    if(mPreloadedBitmaps.count(path) == 0)
    {
        mPreloadedBitmaps[path] = std::move(readBitmapFromFile(path));

    }

    if(pRenderContext) {
        pRenderContext->updateTextureData(tex.get(), (const void*)mPreloadedBitmaps[path]->getData());
    }
}

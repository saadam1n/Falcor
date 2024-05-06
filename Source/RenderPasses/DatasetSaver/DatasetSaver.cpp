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
#include "DatasetSaver.h"

#include <filesystem>
#include <fstream>

namespace {

    const std::string kInputReference = "Reference";
    const std::string kInputColor = "Color";

    const std::string kInputAlbedo = "Albedo";
    const std::string kInputEmission = "Emission";
    const std::string kInputWorldPosition = "WorldPosition";
    const std::string kInputWorldNormal = "WorldNormal";
    const std::string kInputPositionNormalFwidth = "PositionNormalFwidth";
    const std::string kInputLinearZ = "LinearZ";
    const std::string kInputMotionVec = "MotionVec";

    // we need this because falcor will not execute passes that don't have any marked outputs
    const std::string kOutputDummy = "dummyOut";

    int screenWidth = 1920;
    int screenHeight = 1080;

    const char* kDictSamplesPerPixel = "samplesPerPixel";

    const char* kTargetSamples = "targetSamples";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, DatasetSaver>();
}

DatasetSaver::DatasetSaver(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice), currentStorageIndex(0), mTargetSamples(1024 * 16)
{
    for (const auto& [key, value] : props)
    {
        if (key == kTargetSamples)
            mTargetSamples = value;
        else
            logWarning("Unknown property '{}' in DatasetSaver properties.", key);
    }


    setStoragePath("C:/FalcorFiles/Dataset0");

    tempDownloadTexture =  make_ref<Texture>(pDevice, Resource::Type::Texture2D, ResourceFormat::RGBA32Float, screenWidth, screenHeight,  1, 1, 1, 1, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource, nullptr);
    tempDownloadCpuBuffer.resize(screenWidth*screenHeight);
}

Properties DatasetSaver::getProperties() const
{
    return {};
}

RenderPassReflection DatasetSaver::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;

    reflector.addInput(kInputReference, "ground truth image");
    reflector.addInput(kInputColor, "image generated in real time w/ 1 spp");
    reflector.addInput(kInputAlbedo, "albedo input from path tracer");
    reflector.addInput(kInputEmission, "emission input from gbuffer");
    reflector.addInput(kInputWorldPosition, "world pos input from gbuffer");
    reflector.addInput(kInputWorldNormal, "world normal input from gbuffer");
    reflector.addInput(kInputPositionNormalFwidth, "idk something from gbuffer");
    reflector.addInput(kInputLinearZ, "depth buffer from gbuffer");
    reflector.addInput(kInputMotionVec, "motion vec from gbuffer");

    reflector.addOutput(kOutputDummy, "dummy output");

    return reflector;
}

void DatasetSaver::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    Dictionary& dict = renderData.getDictionary();
    int sampleCount =  dict.samplesPerPixel;

    std::cout << "Current sample count is " << sampleCount << std::endl;

    if (pScene && sampleCount == mTargetSamples)
    {
        setStorageKey(std::to_string(currentStorageIndex++));

        storeImage(pRenderContext, renderData, kInputReference);
        storeImage(pRenderContext, renderData, kInputColor);
        storeImage(pRenderContext, renderData, kInputAlbedo);
        storeImage(pRenderContext, renderData, kInputEmission);
        storeImage(pRenderContext, renderData, kInputWorldPosition);
        storeImage(pRenderContext, renderData, kInputWorldNormal);
        storeImage(pRenderContext, renderData, kInputPositionNormalFwidth);
        storeImage(pRenderContext, renderData, kInputLinearZ);
        storeImage(pRenderContext, renderData, kInputMotionVec);
    }
}

void DatasetSaver::renderUI(Gui::Widgets& widget) {}

void DatasetSaver::setStoragePath(const std::string& path)
{
    std::filesystem::create_directories(path);
    storagePath = path;
}

void DatasetSaver::setStorageKey(const std::string& key)
{
    currentStorgeKey = key;
}

void DatasetSaver::storeImage(RenderContext* pRenderContext, const RenderData& renderData, const std::string& name)
{
    ref<Texture> pTexture = renderData.getTexture(name);

    pRenderContext->blit(pTexture->getSRV(), tempDownloadTexture->getRTV());

    std::string writePath = storagePath + "/" + currentStorgeKey + "-" + name + ".exr";

    tempDownloadTexture->captureToFile(0, 0, writePath, Falcor::Bitmap::FileFormat::ExrFile, Falcor::Bitmap::ExportFlags::None, false);
}

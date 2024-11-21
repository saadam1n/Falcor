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
#include "NeuralNoiseReduction.h"

namespace
{
const char* kSimpleKernelInput = "src";
const char* kSimpleKernelOutput = "dst";
const char* kSimpleKernelShader = "RenderPasses/NeuralNoiseReduction/SimpleKernel.ps.slang";

} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, NeuralNoiseReduction>();
}

void blitTextures(RenderContext* pRenderContext, ref<Texture> src, ref<Texture> dst)
{
    pRenderContext->blit(src->getSRV(), dst->getRTV());
}

NeuralNoiseReduction::NeuralNoiseReduction(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    torch::jit::Module module;
    try
    {
        module = torch::jit::load("C:/FalcorFiles/Models/SimpleKernel4.pt");
    }
    catch (const c10::Error& e)
    {
        std::cerr << "Error loading the model: " << e.what() << "\n";
        throw e;
    }

    std::cout << "Model loaded successfully.\n";

    auto parameters = module.named_parameters();
    torch::Tensor conv1;
    for (const auto& param : parameters)
    {
        std::cout << "Parameter name: " << param.name << "\n";
        std::cout << "Parameter size: " << param.value.sizes() << "\n";

        if (param.name == "conv1.weight")
        {
            conv1 = param.value;
        }
    }

    mpBlurFilter = FullScreenPass::create(mpDevice, kSimpleKernelShader);

    for (int i = 0; i < 13; i++)
    {
        for (int j = 0; j < 13; j++)
        {
            float avg = 0.0f;
            for (int k = 0; k < 3; k++)
            {
                for (int l = 0; l < 3; l++)
                {
                    avg += conv1[k][l][i][j].item<float>();
                }
            }
            avg /= 3.0f;
            mKernel[i][j] = avg;
        }
    }
    
}

void NeuralNoiseReduction::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    allocateFbos(compileData.defaultTexDims, pRenderContext);
}

void NeuralNoiseReduction::allocateFbos(uint2 dim, RenderContext* pRenderContext)
{
    {
        Fbo::Desc desc;
        desc.setSampleCount(0);
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float);
        mpBlurringFbo = Fbo::create2D(mpDevice, dim.x, dim.y, desc);
    }
}

Properties NeuralNoiseReduction::getProperties() const
{
    return {};
}

RenderPassReflection NeuralNoiseReduction::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    reflector.addOutput("dst", "ABCD");
    reflector.addInput("src", "EFGH");
    return reflector;
}


void NeuralNoiseReduction::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // renderData holds the requested resources
    auto pSrc = renderData.getTexture("src");
    auto pDst = renderData.getTexture("dst");

    auto perImageCB = mpBlurFilter->getRootVar()["PerImageCB"];

    perImageCB["src"] = pSrc;
    perImageCB["kernel"].setBlob(mKernel);

    mpBlurFilter->execute(pRenderContext, mpBlurringFbo);

    blitTextures(pRenderContext, mpBlurringFbo->getColorTexture(0), pDst);
}

void NeuralNoiseReduction::renderUI(Gui::Widgets& widget) {}


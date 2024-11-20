#pragma once

#include "Falcor.h"
#include "RenderGraph/RenderPass.h"

// will likely need this
#include "ShaderUtils.h"

#include "ParameterReflecter.h"

#include "TextureData.h"
#include "TextureReflecter.h"

using namespace Falcor;

class RenderingComponent : public Object
{
public:
    RenderingComponent(ref<Device> pDevice);
    virtual ~RenderingComponent() = default;

    virtual void reflectTextures(TextureReflecter& reflecter) = 0;
    virtual void reflectParameters(ParameterReflecter& reflecter) = 0;

    virtual void forward(RenderContext* pRenderContext, const TextureData& textureData) = 0;
    virtual void backward(RenderContext* pRenderContext, const TextureData& textureData) = 0;

protected:
    ref<Device> mpDevice;
};

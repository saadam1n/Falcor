#pragma once

#include "Falcor.h"
#include "RenderGraph/RenderPass.h"

#include "RenderingComponent.h"

using namespace Falcor;

#include <map>

// This basically connects rendering components to their inputs/outputs
class SubrenderGraph : public Object
{
public:
    SubrenderGraph(ref<Device> device);
    // reflect a texture
    void registerComponent(ref<RenderingComponent> component);

    // connect two components together
    void createEdge(
        ref<RenderingComponent> srcComponent,
        const std::string& srcName,
        ref<RenderingComponent> dstComponent,
        const std::string& dstName,
        bool dynamicEntry = false
    );

    // connect a component directly a texture
    void createEdge(ref<RenderingComponent> srcComponent, const std::string& srcName, ref<Texture> dstTexture, bool dynamicEntry = false);
    void createEdge(ref<Texture> srcTexture, ref<RenderingComponent> dstComponent, const std::string& dstName, bool dynamicEntry = false);

    void loadDataAndExecForward(RenderContext* pRenderContext, ref<RenderingComponent> component);
    void loadDataAndExecBackward(RenderContext* pRenderContext, ref<RenderingComponent> component);

    ref<Texture> getVertex(ref<RenderingComponent> component, const std::string& name);

private:
    ref<Device> mpDevice;

    void throwIfNotInMap(ref<RenderingComponent> component);

    std::map<ref<RenderingComponent>, TextureData> mComponentDataTable;
};

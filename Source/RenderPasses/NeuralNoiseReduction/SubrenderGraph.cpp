#include "Subrendergraph.h"

SubrenderGraph::SubrenderGraph(ref<Device> device) : mpDevice(device) {}

void SubrenderGraph::registerComponent(ref<RenderingComponent> component)
{
    TextureReflecter reflecter;

    component->reflect(reflecter);

    TextureData data;

    // for each output, we need to create a texture
    // the idea is that a component is in control of its own outputs
    // this works well except for the the final output texture (which will need to be deallocated)
    const auto& outputs = reflecter.getOutputs();
    for (const auto& out : outputs)
    {
        ref<Texture> tex = mpDevice->createTexture2D(out.width, out.height, out.fmt, 1, 1, nullptr, out.flags);
        data.setTexture(out.name, tex);
    }

    // now we just set all other textures to null for the moment
    const auto& inputs = reflecter.getInputs();
    for (const auto& in : inputs)
    {
        data.setTexture(in.name, nullptr);
    }

    // now that data is set, we store it in the map
    mComponentDataTable[component] = data;
}

void SubrenderGraph::createEdge(
    ref<RenderingComponent> srcComponent,
    const std::string& srcName,
    ref<RenderingComponent> dstComponent,
    const std::string& dstName
)
{
    throwIfNotInMap(srcComponent);
    throwIfNotInMap(dstComponent);

    // basically we need to set the dst output to the input
    mComponentDataTable[dstComponent].setTexture(dstName, mComponentDataTable[srcComponent].getTexture(srcName), true);
}

void SubrenderGraph::createEdge(ref<RenderingComponent> srcComponent, const std::string& srcName, ref<Texture> dstTexture)
{
    std::cout << "CAUTION: Attemption to set " << srcName
              << " of a component to point towards a texture. This effectively sets the output location to the texture!" << std::endl;

    throwIfNotInMap(srcComponent);

    mComponentDataTable[srcComponent].setTexture(srcName, dstTexture, true);
}

void SubrenderGraph::createEdge(ref<Texture> srcTexture, ref<RenderingComponent> dstComponent, const std::string& dstName)
{
    throwIfNotInMap(dstComponent);

    mComponentDataTable[dstComponent].setTexture(dstName, srcTexture, true);
}

void SubrenderGraph::loadDataAndExecForward(RenderContext* pRenderContext, ref<RenderingComponent> component)
{
    throwIfNotInMap(component);
    component->forward(pRenderContext, mComponentDataTable[component]);
}

void SubrenderGraph::loadDataAndExecBackward(RenderContext* pRenderContext, ref<RenderingComponent> component)
{
    // TODO: backward needs a backprop render graph, which is NYI
}

void SubrenderGraph::throwIfNotInMap(ref<RenderingComponent> component)
{
    if (mComponentDataTable.count(component) == 0)
    {
        FALCOR_THROW("Render component not added to render subgraph!");
    }
}

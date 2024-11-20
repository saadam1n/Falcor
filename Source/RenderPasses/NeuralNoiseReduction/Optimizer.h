#pragma once

#include "Falcor.h"
#include "RenderGraph/RenderPass.h"

#include "RenderingComponent.h"
#include "ParameterReflecter.h"

using namespace Falcor;





// the goal of this class is to:
// 1) give components an area to dump their derivatives for parameters and (in the future) textures that are passed between textures
// 2) automatically optimize their parameters based off these derivatives
//
// we assume that all parameters are float values (or some form of float array)
//
class Optimizer : public Object
{
public:
    Optimizer(ref<Device> pDevice) : mpDevice(pDevice) {}

    void registerComponent(ref<RenderingComponent> component);

private:
    ref<Device> mpDevice;

    std::vector<RuntimeParameter> mParameters;
};

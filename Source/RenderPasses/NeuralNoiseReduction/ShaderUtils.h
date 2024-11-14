#pragma once

#include "Falcor.h"
#include "Core/Pass/FullScreenPass.h"
#include "Core/Pass/ComputePass.h"

using namespace Falcor;

enum NeuralNetPassType
{
    NEURAL_NET_PASS_TYPE_NONE, // for stuff that isn't part of a nestwork
    NEURAL_NET_PASS_TYPE_FORWARD,
    NEURAL_NET_PASS_TYPE_BACKWARD,
};

ref<FullScreenPass> createFullscreenPassAndDumpIR(
    ref<Device> pDevice,
    const std::string& path,
    NeuralNetPassType nnpt = NEURAL_NET_PASS_TYPE_NONE,
    const DefineList& dl = DefineList()
);

ref<ComputePass> createComputePassAndDumpIR(
    ref<Device> pDevice,
    const std::string& path,
    NeuralNetPassType nnpt = NEURAL_NET_PASS_TYPE_NONE,
    const DefineList& dl = DefineList()
);

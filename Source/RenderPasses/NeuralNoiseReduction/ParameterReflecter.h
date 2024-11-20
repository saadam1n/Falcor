#pragma once

#include "Falcor.h"
#include "RenderGraph/RenderPass.h"

#include "Common.h"

using namespace Falcor;

#include <vector>

template<typename T>
struct LearnableParameter
{
public:
    T var;
    ref<Texture> accum;
};

struct RuntimeParameter
{
    // for debug... usually
    std::string name;

    // number of 4-byte floats we need to optimize
    int numElements;

    // address of parameter to be optimized, memory is managed externally
    float* var;

    // place where derivatives are accumulated
    // it is a pointer because it points towards the location of the paramter's texture
    ref<Texture>* accum;
};

#define REGISTER_PARAMETER(reflecter, param) reflecter.registerParameter(param, #param)

class ParameterReflecter
{
public:
    template<typename T>
    void registerParameter(LearnableParameter<T>& param, const std::string& name = "")
    {
        // we need to convert this from a template type to a "runtime" type
        // i.e. basically perform type erasure somehow
        // for the runtime what we need is:
        // 1) the location of the values to optimize
        // 2) a place to store and update associated values in more complex optimizers like Adam

        RuntimeParameter rt;
        rt.name = name;
        rt.numElements = sizeof(T) / sizeof(float);
        rt.var = (float*)&param.var;
        rt.accum = &param.accum;

        // we need to create an accumulation location for this variable
        // ideally we would have a system that reuses the same block of memory for everything but right now simple will do
        ref<Texture> accum = mpDevice->createTexture2D(
            sParams.patchWidth,
            sParams.patchHeight,
            ResourceFormat::R32Float, // 1 float per slice/element in the array
            rt.numElements,
            1,
            nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );

        mParameters.push_back(rt);
    }

    std::vector<RuntimeParameter>& getParameters() { return mParameters; }
private:
    std::vector<RuntimeParameter> mParameters;
};

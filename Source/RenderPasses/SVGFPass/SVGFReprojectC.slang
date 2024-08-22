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
import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;
import SVGFCommon;

// Workaround for isnan() not working in slang.
bool isNaN(float f)
{
    uint u = asuint(f) & ~0x80000000u; // clear out the sign bit
    return (u > 0x7F800000);           // greater than Inf is NaN
}

#define MLP_NUM_WEIGHTS 1192
#define NUM_FILTER_ITERATIONS 5

cbuffer PerImageCB
{
    Texture2D gMotion;
    Texture2D gPositionNormalFwidth;
    Texture2D gColor;
    Texture2D gAlbedo;
    Texture2D gEmission;
    Texture2D gPrevIllum;
    Texture2D gPrevTemporalAccum;
    Texture2D gPrevMoments;
    Texture2D gLinearZAndNormal;
    Texture2D gPrevLinearZAndNormal;
    Texture2D gPrevHistoryLength;
    Texture2D gPrevFiltered[NUM_FILTER_ITERATIONS];

    float       dvAlpha;
    float       dvMomentsAlpha;

    float3 dvLuminanceParams;
    float dvReprojKernel[3];
    float dvReprojParams[4];

    float dvMlpWeights[MLP_NUM_WEIGHTS];

    RWByteAddressBuffer daIllumination;
    RWByteAddressBuffer daMoments;
    RWByteAddressBuffer daHistoryLength;

    RWByteAddressBuffer daMlpWeights;
};

static const float QUANTIZATION = 1.0;

no_diff
float3 getAlbedo(int2 ipos)
{
    return gAlbedo[ipos].rgb;
}

float3 getColor(int2 ipos)
{
    return gColor[ipos].rgb;
}

float3 getEmission(int2 ipos)
{
    return gEmission[ipos].rgb;
}

no_diff
float3 getNonemissiveColor(int2 ipos) // crude workaround for slang bug
{
    return getColor(ipos) - getEmission(ipos);
}

no_diff
float4 getLinearZAndNormal(int2 ipos)
{
    return gLinearZAndNormal[ipos];
}

no_diff
float4 getPrevLinearZAndNormal(int2 ipos)
{
    return gPrevLinearZAndNormal[ipos];
}

no_diff
float2 getMotion(int2 ipos)
{
    return gMotion[ipos].rg;
}

no_diff
float getPositionNormalFwidth(int2 ipos)
{
    return gPositionNormalFwidth[ipos].r;
}

// ===============================
// Stuff we want to actually accum
// ===============================

// Bilinear code

int getBilinearIndex(int2 offset)
{
    int storageIdx = offset.y * 2 + offset.x;
    return storageIdx;
}

void d_getPrevIllumBilinear(int2 p, int2 offset, float4 grad)
{
    // storeDerivBuf4(<need own buffer>, p, grad, gAlbedo, getBilinearIndex(offset));
}

[BackwardDerivative(d_getPrevIllumBilinear)]
float4 getPrevIllumBilinear(int2 p, int2 offset)
{
    return gPrevIllum[p];
}

void d_getPrevTemporalAccumBilinear(int2 p, int2 offset, float4 grad)
{
    //storeDerivBuf4(daIllumination, p, grad, gAlbedo, getBilinearIndex(offset));
}

[BackwardDerivative(d_getPrevTemporalAccumBilinear)]
float4 getPrevTemporalAccumBilinear(int2 p, int2 offset)
{
    return gPrevTemporalAccum[p];
}

void d_getPrevMomentsBilinear(int2 p, int2 offset, float2 grad)
{
    //storeDerivBuf4(daMoments, p, float4(grad, 0.0f.xx), gAlbedo, getBilinearIndex(offset));
}

[BackwardDerivative(d_getPrevMomentsBilinear)]
float2 getPrevMomentsBilinear(int2 p, int2 offset)
{
    return gPrevMoments[p].rg;
}

// Bilteral code

int getBilateralIndex(int2 offset)
{
    int storageIdx = (offset.y + 1) * 3 + (offset.x + 1);
    return storageIdx + 4;
}

void d_getPrevIllumBilateral(int2 p, int2 offset, float4 grad)
{
    // storeDerivBuf4(<need own buffer>, p, grad, gAlbedo, getBilateralIndex(offset));
}

[BackwardDerivative(d_getPrevIllumBilateral)]
float4 getPrevIllumBilateral(int2 p, int2 offset)
{
    return gPrevIllum[p];
}

void d_getPrevTemporalAccumBilateral(int2 p, int2 offset, float4 grad)
{
    //storeDerivBuf4(daIllumination, p, grad, gAlbedo, getBilateralIndex(offset));
}

[BackwardDerivative(d_getPrevTemporalAccumBilateral)]
float4 getPrevTemporalAccumBilateral(int2 p, int2 offset)
{
    return gPrevTemporalAccum[p];
}

void d_getPrevMomentsBilateral(int2 p, int2 offset, float2 grad)
{
    //storeDerivBuf4(daMoments, p, float4(grad, 0.0f.xx), gAlbedo, getBilateralIndex(offset));
}

[BackwardDerivative(d_getPrevMomentsBilateral)]
float2 getPrevMomentsBilateral(int2 p, int2 offset)
{
    return gPrevMoments[p].rg;
}

// History length

void d_getPrevHistoryLength(int2 ipos, float grad)
{
    //storeDerivBuf4(daHistoryLength, ipos, float4(grad, 0.0f.xxx), gAlbedo);
}

[BackwardDerivative(d_getPrevHistoryLength)]
float getPrevHistoryLength(int2 ipos)
{
    return gPrevHistoryLength[ipos].r;
}

// Other thing I might want to accum eventually
no_diff float4 getPrevFiltered(int2 ipos, int idx)
{
    return gPrevFiltered[idx][ipos];
}

// todo: get a differentiable version of this
bool isReprjValid(int2 coord, float Z, float Zprev, float fwidthZ, float3 normal, float3 normalPrev, float fwidthNormal, float[4] reprojectionParamters)
{
    const int2 imageDim = getTextureDims(gColor, 0);

    // check whether reprojected pixel is inside of the screen
    if (any(coord < int2(1, 1)) || any(coord > imageDim - int2(1, 1)))
        return false;

    // check if deviation of depths is acceptable
    if (abs(Zprev - Z) / (fwidthZ + 1e-2f) > reprojectionParamters[2]) return false;

    // check normals for compatibility
    if (distance(normal, normalPrev) / (fwidthNormal + 1e-2) > reprojectionParamters[3]) return false;

    return true;
}

struct PrevFrameData : IDifferentiable
{
    float4 prevTemporalAccum;
    float4 prevIllum;
    float4 prevFiltered[NUM_FILTER_ITERATIONS];
    float2 prevMoments;
    float historyLength;
    bool success;
};

void d_bilinearLoad(float2 posPrev, int2 iposPrev, float2 depth, float3 normal, float normalFwidth, inout DifferentialPair<float[4]> dpArgReprojectionParamters, PrevFrameData.Differential grad)
{
    const int2 offset[4] = { int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1) };

    float sumw = 0;
    float x = frac(posPrev.x);
    float y = frac(posPrev.y);

    // bilinear weights
    const float w[4] = { (1 - x) * (1 - y), x * (1 - y), (1 - x) * y, x * y };

    // check for all 4 taps of the bilinear filter for validity
    bool v[4];
    bool success = false;
    for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++)
    {
        int2 loc = int2(posPrev) + offset[sampleIdx];
        float2 depthPrev = getPrevLinearZAndNormal(loc).xy;
        float3 normalPrev = oct_to_ndir_snorm(getPrevLinearZAndNormal(loc).zw);

        v[sampleIdx] = isReprjValid(iposPrev, depth.x, depthPrev.x, depth.y, normal, normalPrev, normalFwidth, dpArgReprojectionParamters.p);

        success = success || v[sampleIdx];
        if (v[sampleIdx])
        {
            sumw += w[sampleIdx];
        }
    }

    if (success)
    {
        // not needed since we don't have a differentaible isReprojValid func
        // float4 dLdSumWFromIllum = -grad.prevIllum / (sumw * sumw);
        // float2 dLdSumWFromMoments = -grad.prevMoments / (sumw * sumw);
        // float dLdSumW = dot(dLdSumWFromIllum, 1.0f.xxxx) + dot(dLdSumWFromMoments, 1.0f.xx);

        float4 dLdPrevTemporalAccum = -grad.prevTemporalAccum / sumw;
        float4 dLdPrevIllum = -grad.prevIllum / sumw;
        float2 dLdPrevMoments = -grad.prevMoments / sumw;

        for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++)
        {
            int2 loc = int2(posPrev) + offset[sampleIdx];
            if (v[sampleIdx])
            {
                __bwd_diff(getPrevTemporalAccumBilinear)(loc, offset[sampleIdx], dLdPrevTemporalAccum);
                __bwd_diff(getPrevIllumBilinear)(loc, offset[sampleIdx], dLdPrevIllum);
                __bwd_diff(getPrevMomentsBilinear)(loc, offset[sampleIdx], dLdPrevMoments);
            }
        }
    }
}

[BackwardDerivative(d_bilinearLoad)]
PrevFrameData bilinearLoad(no_diff float2 posPrev, int2 iposPrev, no_diff float2 depth, no_diff float3 normal, no_diff float normalFwidth, float[4] reprojectionParamters)
{
    const int2 offset[4] = { int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1) };

    PrevFrameData pfd;

    pfd.prevTemporalAccum = float4(0, 0, 0, 0);
    pfd.prevIllum = float4(0, 0, 0, 0);
    pfd.prevMoments = float2(0, 0);
    pfd.success = false;

    for (int i = 0; i < NUM_FILTER_ITERATIONS; i++)
    {
        pfd.prevFiltered[i] = float4(0, 0, 0, 0);
    }

    // check for all 4 taps of the bilinear filter for validity
    bool v[4];
    for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++)
    {
        int2 loc = int2(posPrev) + offset[sampleIdx];
        float2 depthPrev = getPrevLinearZAndNormal(loc).xy;
        float3 normalPrev = oct_to_ndir_snorm(getPrevLinearZAndNormal(loc).zw);

        v[sampleIdx] = isReprjValid(iposPrev, depth.x, depthPrev.x, depth.y, normal, normalPrev, normalFwidth, reprojectionParamters);

        pfd.success = pfd.success || v[sampleIdx];
    }

    if (pfd.success)
    {
        float sumw = 0;
        float x = frac(posPrev.x);
        float y = frac(posPrev.y);

        // bilinear weights
        const float w[4] = { (1 - x) * (1 - y), x * (1 - y), (1 - x) * y, x * y };

        // perform the actual bilinear interpolation
        for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++)
        {
            const int2 loc = int2(posPrev) + offset[sampleIdx];
            if (v[sampleIdx])
            {
                pfd.prevTemporalAccum += w[sampleIdx] * getPrevTemporalAccumBilinear(loc, offset[sampleIdx]);
                pfd.prevIllum += w[sampleIdx] * getPrevIllumBilinear(loc, offset[sampleIdx]);
                pfd.prevMoments += w[sampleIdx] * getPrevMomentsBilinear(loc, offset[sampleIdx]);
                sumw += w[sampleIdx];

                for (int i = 0; i < NUM_FILTER_ITERATIONS; i++)
                {
                    float4 f = getPrevFiltered(loc, i);
                    pfd.prevFiltered[i] += w[sampleIdx] * f;
                }
            }
        }

        // redistribute weights in case not all taps were used
        pfd.success = (sumw >= 0.01);
        pfd.prevTemporalAccum /= sumw;
        pfd.prevIllum /= sumw;
        pfd.prevMoments /= sumw;

        for (int i = 0; i < NUM_FILTER_ITERATIONS; i++)
        {
            pfd.prevFiltered[i] /= sumw;
        }
    }

    return pfd;
}

[BackwardDifferentiable]
PrevFrameData bilateralLoad(int2 iposPrev, float[3] kernelWeights, no_diff float2 depth, no_diff float3 normal, no_diff float normalFwidth, float[4] reprojectionParamters)
{
    float nValid = 0.0;
    float totalWeight = 0.0;

    PrevFrameData pfd;
    pfd.prevTemporalAccum = float4(0.0f);
    pfd.prevIllum = float4(0.0f);
    pfd.prevMoments = float2(0.0f);

    for (int i = 0; i < NUM_FILTER_ITERATIONS; i++)
    {
        pfd.prevFiltered[i] = float4(0, 0, 0, 0);
    }

    // this code performs a binary descision for each tap of the cross-bilateral filter
    for (int yy = -1; yy <= 1; yy++)
    {
        for (int xx = -1; xx <= 1; xx++)
        {
            const int2 p = iposPrev + int2(xx, yy);
            const float2 depthFilter = getPrevLinearZAndNormal(p).xy;
            const float3 normalFilter = oct_to_ndir_snorm(getPrevLinearZAndNormal(p).zw);
            float curWeight = kernelWeights[abs(yy) + abs(xx)];

            if (isReprjValid(iposPrev, depth.x, depthFilter.x, depth.y, normal, normalFilter, normalFwidth, reprojectionParamters))
            {
                pfd.prevTemporalAccum += curWeight * getPrevTemporalAccumBilateral(p, int2(xx, yy));
                pfd.prevIllum += curWeight * getPrevIllumBilateral(p, int2(xx, yy));
                pfd.prevMoments += curWeight * getPrevMomentsBilateral(p, int2(xx, yy));

                for (int i = 0; i < NUM_FILTER_ITERATIONS; i++)
                {
                    float4 f = getPrevFiltered(p, i);
                    pfd.prevFiltered[i] += curWeight* f;
                }

                nValid += 1.0;
                totalWeight += curWeight;
            }
        }
    }
    if (nValid > 0)
    {
        pfd.success = true;

        totalWeight = max(totalWeight, 1e-6);

        pfd.prevTemporalAccum /= totalWeight;
        pfd.prevIllum /= totalWeight;
        pfd.prevMoments /= totalWeight;

        for (int i = 0; i < NUM_FILTER_ITERATIONS; i++)
        {
            pfd.prevFiltered[i] /= totalWeight;
        }
    }

    return pfd;
}

// get the temporal data (this function does not accumulate)
// [BackwardDifferentiable]
PrevFrameData loadPrevData(no_diff float2 posH, float[4] reprojectionParamters, float[3] kernelWeights)
{
    const int2 ipos = posH;
    const float2 imageDim = float2(getTextureDims(gColor, 0));

    const float2 motion = getMotion(ipos);
    const float normalFwidth = getPositionNormalFwidth(ipos);

    // +0.5 to account for texel center offset
    const int2 iposPrev = int2(float2(ipos) + motion.xy * imageDim + float2(0.5, 0.5));
    const no_diff float2 posPrev = diffFloor(posH.xy) + motion.xy * imageDim;

    float2 depth = getLinearZAndNormal(ipos).xy;
    float3 normal = oct_to_ndir_snorm(getLinearZAndNormal(ipos).zw);

    PrevFrameData pfd = bilinearLoad(posPrev, iposPrev, depth, normal, normalFwidth, reprojectionParamters);

    if (!pfd.success)
    {
        pfd = bilateralLoad(iposPrev, kernelWeights, depth, normal, normalFwidth, reprojectionParamters);
    }

    if (pfd.success)
    {
        // crude, fixme
        pfd.historyLength = getPrevHistoryLength(iposPrev);
    }
    else
    {
        pfd.prevTemporalAccum = float4(0, 0, 0, 0);
        pfd.prevIllum = float4(0, 0, 0, 0);
        pfd.prevMoments = float2(0, 0);
        pfd.historyLength = 0;
    }

    return pfd;
}

// not used currently
float computeVarianceScale(float numSamples, float loopLength, float alpha)
{
    const float aa = (1.0 - alpha) * (1.0 - alpha);
    return (1.0 - pow(aa, min(loopLength, numSamples))) / (1.0 - aa);
}

struct REPROJ_OUT : IDifferentiable
{
    float4 OutIllumination;
    float2 OutMoments;
    float  OutHistoryLength;
    float4 OutTemporalAccum;
};

struct PS_OUT {
    float4 OutIllumination               : SV_TARGET0;
    float2 OutMoments                    : SV_TARGET1;
    float  OutHistoryLength              : SV_TARGET2;
    float4 OutTemporalAccum         : SV_TARGET3;
};




float getInput(float4 curIllum, float4 prevIllum, float4[NUM_FILTER_ITERATIONS] prevFiltered, int i)
{
    if (i < 4)
    {
        return curIllum[i];
    }
    else if(i < 8)
    {
        return prevIllum[i - 4];
    }
    else
    {
        int index = i / 4 - 2;
        return prevFiltered[index][i % 4];
    }
}

#define K_NUM_FEATURES (8 + 4 * NUM_FILTER_ITERATIONS)
#define K_INPUT_LAYER_NODES K_NUM_FEATURES
#define K_HIDDEN_LAYER_NODES 12
#define K_OUTPUT_LAYER_NODES 6

const int getWeightLayer0(const int node, const int source)
{
    return node * (K_NUM_FEATURES + 1) + source;
}

const int getWeightLayer1(const int node, const int source)
{
    return getWeightLayer0(K_INPUT_LAYER_NODES - 1, K_NUM_FEATURES) + 1 + node * (K_INPUT_LAYER_NODES + 1) + source;
}

const int getWeightLayer2(const int node, const int source)
{
    return getWeightLayer1(K_HIDDEN_LAYER_NODES - 1, K_INPUT_LAYER_NODES) + 1 + node * (K_HIDDEN_LAYER_NODES + 1) + source;
}

const int getWeightIndex(const int layer, const int node, const int source)
{
    int idx = 0;
    if (layer == 0)
    {
        idx = getWeightLayer0(node, source);
    }
    else if (layer == 1)
    {
        idx = getWeightLayer1(node, source);
    }
    else
    {
        idx = getWeightLayer2(node, source);
    }

    return idx;
}

static float4 dLdMlpWeights = float4(1.0f);
static int2 storePos;
void d_getWeight(int layer, int node, int source, float grad)
{
    int idx = getWeightIndex(layer, node, source);

    dLdMlpWeights[idx % 4] = grad;

    if (idx % 4 == 0)
    {
        storeDerivBuf4(daMlpWeights, storePos, dLdMlpWeights, gAlbedo, idx / 4);
    }
}

[BackwardDerivative(d_getWeight)]
float getWeight(int layer, int node, int source)
{
    return dvMlpWeights[getWeightIndex(layer, node, source)];
}

#define COLOR_INPUT_POWER 0.2f
void d_ff_mlp(float4 curIllum, inout DifferentialPair<float4> prevIllum, float4[NUM_FILTER_ITERATIONS] prevFiltered, float4 dLoss)
{
    float inputLayerOutputs[K_INPUT_LAYER_NODES];

    for (int i = 0; i < K_INPUT_LAYER_NODES; i++)
    {
        inputLayerOutputs[i] = 0.0f;
        for (int j = 0; j <= K_NUM_FEATURES; j++)
        {
            inputLayerOutputs[i] += getWeight(0, i, j) * (j == K_NUM_FEATURES ? 1.0f : pow(getInput(curIllum, prevIllum.p, prevFiltered, j), COLOR_INPUT_POWER));
        }
        inputLayerOutputs[i] = max(inputLayerOutputs[i], 0.0f);
    }

    float hiddenLayerOutputs[K_HIDDEN_LAYER_NODES];
    for (int i = 0; i < K_HIDDEN_LAYER_NODES; i++)
    {
        hiddenLayerOutputs[i] = 0.0f;
        for (int j = 0; j <= K_INPUT_LAYER_NODES; j++)
        {
            hiddenLayerOutputs[i] += getWeight(1, i, j) * (j == K_INPUT_LAYER_NODES ? 1.0f : inputLayerOutputs[j]);
        }
        hiddenLayerOutputs[i] = max(hiddenLayerOutputs[i], 0.0f);
    }

    float outputLayerOutputs[K_OUTPUT_LAYER_NODES];
    for (int i = 0; i < K_OUTPUT_LAYER_NODES; i++)
    {
        outputLayerOutputs[i] = 0.0f;
        for (int j = 0; j <= K_HIDDEN_LAYER_NODES; j++)
        {
            outputLayerOutputs[i] += getWeight(2, i, j) * (j == K_HIDDEN_LAYER_NODES ? 1.0f : hiddenLayerOutputs[j]);
        }
    }

    float eOut[K_OUTPUT_LAYER_NODES];
    float div = 0.0f;
    for (int i = 0; i < K_OUTPUT_LAYER_NODES; i++)
    {
        eOut[i] = exp(outputLayerOutputs[i]);
        eOut[i] = clamp(eOut[i], 1e-6f, 1e+6f);
        div += eOut[i];
    }

    float weights[K_OUTPUT_LAYER_NODES];
    for (int i = 0; i < K_OUTPUT_LAYER_NODES; i++)
    {
        weights[i] = eOut[i] / div;
    }

    /*
    float4 finalIllum = curIllum * weights[0] + prevIllum * weights[1];
    for (int i = 0; i < NUM_FILTER_ITERATIONS; i++)
    {
        finalIllum += weights[i + 2] * prevFiltered[i];
    }
    */

    float4 dLdFinalIllum = dLoss;
    float4 dLdPrevIllum = dLdFinalIllum * weights[1];

    float dLdWeights[K_OUTPUT_LAYER_NODES];
    dLdWeights[0] = dot(dLdFinalIllum * curIllum, 1.0f.xxxx);
    dLdWeights[1] = dot(dLdFinalIllum * prevIllum.p, 1.0f.xxxx);
    for (int i = 0; i < NUM_FILTER_ITERATIONS; i++)
    {
        dLdWeights[i + 2] = dot(weights[i + 2] * prevFiltered[i], 1.0f.xxxx);
    }

    float dLdDiv = 0.0f;
    for (int i = 0; i < K_OUTPUT_LAYER_NODES; i++)
    {
        dLdDiv += dLdWeights[i] * -eOut[i] / (div * div);
    }

    float dLdOutputLayerOutputs[K_OUTPUT_LAYER_NODES];
    for (int i = 0; i < K_OUTPUT_LAYER_NODES; i++)
    {
        float dLdEout = dLdWeights[i] / div + dLdDiv;
        if (eOut[i] == 1e-6f || eOut[i] == 1e+6f)
        {
            dLdEout = 0.0f;
        }
        dLdOutputLayerOutputs[i] = dLdEout * exp(outputLayerOutputs[i]);
    }


    float dLdHiddenLayerOutputs[K_HIDDEN_LAYER_NODES];
    for (int i = 0; i < K_HIDDEN_LAYER_NODES; i++)
    {
        dLdHiddenLayerOutputs[i] = 0.0f;
    }

    for (int i = K_OUTPUT_LAYER_NODES - 1; i >= 0; i--)
    {
        for (int j = K_HIDDEN_LAYER_NODES; j >= 0; j--)
        {
            float readVal = (j == K_HIDDEN_LAYER_NODES ? 1.0f : hiddenLayerOutputs[j]);

            if (j != K_HIDDEN_LAYER_NODES)
            {
                dLdHiddenLayerOutputs[j] += dLdOutputLayerOutputs[i] * getWeight(2, i, j);
            }

            __bwd_diff(getWeight)(2, i, j, dLdOutputLayerOutputs[i] * readVal);
        }
    }

    // relu
    for (int i = 0; i < K_HIDDEN_LAYER_NODES; i++)
    {
        if (hiddenLayerOutputs[i] == 0.0f)
        {
            dLdHiddenLayerOutputs[i] = 0.0f;
        }
    }

    float dLdInputLayerOutputs[K_INPUT_LAYER_NODES];
    for (int i = 0; i < K_INPUT_LAYER_NODES; i++)
    {
        dLdInputLayerOutputs[i] = 0.0f;
    }

    for (int i = K_HIDDEN_LAYER_NODES - 1; i >= 0; i--)
    {
        for (int j = K_INPUT_LAYER_NODES; j >= 0; j--)
        {
            float readVal = (j == K_INPUT_LAYER_NODES ? 1.0f : inputLayerOutputs[j]);
            // hiddenLayerOutputs[i] += getWeight(1, i, j) * (j == 8 ? 1.0f : inputLayerOutputs[j]);
            if (j != K_INPUT_LAYER_NODES)
            {
                dLdInputLayerOutputs[j] += dLdHiddenLayerOutputs[i] * getWeight(1, i, j);
            }

            __bwd_diff(getWeight)(1, i, j, dLdHiddenLayerOutputs[i] * readVal);
        }
    }

    // Relu
    for (int i = 0; i < K_INPUT_LAYER_NODES; i++)
    {
        if (inputLayerOutputs[i] == 0.0f)
        {
            dLdInputLayerOutputs[i] = 0.0f;
        }
    }

    for (int i = K_INPUT_LAYER_NODES - 1; i >= 0; i--)
    {
        for (int j = K_NUM_FEATURES; j >= 0; j--)
        {
            float readVal = (j == K_NUM_FEATURES ? 1.0f : getInput(curIllum, prevIllum.p, prevFiltered, j));
            // inputLayerOutputs[i] += getWeight(0, i, j) * (j == 8 ? 1.0f : getInput(curIllum, prevIllum.p, j));

            if (false && j >= 4)
            {
                float rawLoss = dLdInputLayerOutputs[i] * getWeight(0, i, j);
                dLdPrevIllum[j - 4] += rawLoss * COLOR_INPUT_POWER * pow(prevIllum.p[j - 4], COLOR_INPUT_POWER - 1.0f);
            }
            __bwd_diff(getWeight)(0, i, j, dLdInputLayerOutputs[i] * readVal);
        }
    }

    prevIllum = diffPair(prevIllum.p, dLdPrevIllum);
}

[BackwardDerivative(d_ff_mlp)]
float4 ff_mlp(no_diff float4 curIllum, float4 prevIllum, no_diff float4[NUM_FILTER_ITERATIONS] prevFiltered)
{
    float inputLayerOutputs[K_INPUT_LAYER_NODES];

    for (int i = 0; i < K_INPUT_LAYER_NODES; i++)
    {
        inputLayerOutputs[i] = 0.0f;
        for (int j = 0; j <= K_NUM_FEATURES; j++)
        {
            inputLayerOutputs[i] += getWeight(0, i, j) * (j == K_NUM_FEATURES ? 1.0f : pow(getInput(curIllum, prevIllum, prevFiltered, j), COLOR_INPUT_POWER));
        }
        inputLayerOutputs[i] = max(inputLayerOutputs[i], 0.0f);
    }

    float hiddenLayerOutputs[K_HIDDEN_LAYER_NODES];
    for (int i = 0; i < K_HIDDEN_LAYER_NODES; i++)
    {
        hiddenLayerOutputs[i] = 0.0f;
        for (int j = 0; j <= K_INPUT_LAYER_NODES; j++)
        {
            hiddenLayerOutputs[i] += getWeight(1, i, j) * (j == K_INPUT_LAYER_NODES ? 1.0f : inputLayerOutputs[j]);
        }
        hiddenLayerOutputs[i] = max(hiddenLayerOutputs[i], 0.0f);
    }

    float outputLayerOutputs[K_OUTPUT_LAYER_NODES];
    for (int i = 0; i < K_OUTPUT_LAYER_NODES; i++)
    {
        outputLayerOutputs[i] = 0.0f;
        for (int j = 0; j <= K_HIDDEN_LAYER_NODES; j++)
        {
            outputLayerOutputs[i] += getWeight(2, i, j) * (j == K_HIDDEN_LAYER_NODES ? 1.0f : hiddenLayerOutputs[j]);
        }
    }

    float eOut[K_OUTPUT_LAYER_NODES];
    float div = 0.0f;
    for (int i = 0; i < K_OUTPUT_LAYER_NODES; i++)
    {
        eOut[i] = exp(outputLayerOutputs[i]);
        eOut[i] = clamp(eOut[i], 1e-6f, 1e+6f);
        div += eOut[i];
    }

    float weights[K_OUTPUT_LAYER_NODES];
    for (int i = 0; i < K_OUTPUT_LAYER_NODES; i++)
    {
        weights[i] = eOut[i] / div;
    }



    float4 finalIllum = curIllum * weights[0] + prevIllum * weights[1];
    for (int i = 0; i < NUM_FILTER_ITERATIONS; i++)
    {
        finalIllum += weights[i + 2] * prevFiltered[i];
    }

    return finalIllum;
}


// todo: back prop history length
void d_reproject( FullScreenPassVsOut vsOut, inout DifferentialPair<float3> luminanceParams, inout DifferentialPair<float[4]> reprojectionParamters,
    inout DifferentialPair<float[3]> kernelWeights, inout DifferentialPair<float> alpha, inout DifferentialPair<float> momentsAlpha, REPROJ_OUT.Differential dLoss)
{
    const float4 posH = vsOut.posH;
    const int2 ipos = posH.xy;

    float3 illumination = demodulate(getNonemissiveColor(ipos), getAlbedo(ipos));

    // Workaround path tracer bugs. TODO: remove this when we can.
    if (isNaN(illumination.x) || isNaN(illumination.y) || isNaN(illumination.z))
    {
        illumination = float3(0, 0, 0);
    }

    var pfd = loadPrevData(posH.xy, reprojectionParamters.p, kernelWeights.p); // get temporal data with weights

    float2 prevMoments = pfd.prevMoments;
    float historyLength = pfd.historyLength;
    bool success = pfd.success;

    historyLength = min(reprojectionParamters.p[0], success ? historyLength + 1.0f : 1.0f);

    // it appears that the temporal accumulation does not have a constant alpha, instead it is adaptive

    // this adjusts the alpha for the case where insufficient history is available.
    // It boosts the temporal accumulation to give the samples equal weights in
    // the beginning.
    const float alphaAdjusted = success ? max(alpha.p, reprojectionParamters.p[1] / historyLength) : 1.0;
    const float alphaMomentsAdjusted = success ? max(momentsAlpha.p, reprojectionParamters.p[1] / historyLength) : 1.0;

    // compute first two moments of luminance
    float2 moments;
    moments.r = dot(illumination, luminanceParams.p.rgb);
    moments.g = moments.r * moments.r;

    float2 pm = moments;

    // temporal integration of the moments
    moments = lerp(prevMoments, moments, alphaMomentsAdjusted);

    float variance = max(0.f, moments.g - moments.r * moments.r);

    float4 initialEstimate = lerp(pfd.prevTemporalAccum, float4(illumination, 0), alphaAdjusted);
    DifferentialPair<float4> dLdPrevIllum = diffPair(pfd.prevIllum);
    float4 dLdIllumination = float4(dLoss.OutIllumination.rgb, 0.0f);
    if (false && success)
    {
        storePos = ipos;
        __bwd_diff(ff_mlp)(initialEstimate, dLdPrevIllum, pfd.prevFiltered, dLdIllumination);
    }



    float dLdVariance = dLoss.OutIllumination.a;
    if (variance == 0.0f)
    {
        dLdVariance = 0.0f;
    }

    float2 dLdMoments = dLoss.OutMoments;
    dLdMoments.g += dLdVariance;
    dLdMoments.r += dLdVariance * -2.0f * moments.r;
    float2 dLdOriginalMoments = dLdMoments * alphaMomentsAdjusted;
    float2 dLdPrevMoments = dLdMoments * -alphaMomentsAdjusted;

    dLdOriginalMoments.r += dLdOriginalMoments.g * 2.0f * pm.r;

    luminanceParams = diffPair(luminanceParams.p, dLdOriginalMoments.r * illumination);

    float dLdHistoryLength = 0.0f;

    float dLdMomentsAlphaAdjusted = dot(dLdMoments * (pm - prevMoments), 1.0f.xx);

    float dLdMomentsALpha = 0.0f;
    if (success && alphaMomentsAdjusted > reprojectionParamters.p[1] / historyLength)
    {
        dLdMomentsALpha = dLdMomentsAlphaAdjusted;
    }
    else
    {
        dLdHistoryLength += dLdMomentsAlphaAdjusted * -reprojectionParamters.p[1] / (historyLength * historyLength);
    }
    momentsAlpha = diffPair(momentsAlpha.p, dLdMomentsALpha);

    float dLdAlphaAdjusted = dot(dLdIllumination * (initialEstimate - pfd.prevTemporalAccum), 1.0f.xxxx);
    float dLdAlpha = 0.0f;
    if (success && alphaAdjusted > reprojectionParamters.p[1] / historyLength)
    {
        dLdAlpha = dLdAlphaAdjusted;
    }
    else
    {
        dLdHistoryLength += dLdAlphaAdjusted * -reprojectionParamters.p[1] / (historyLength * historyLength);
    }
    alpha = diffPair(alpha.p, dLdAlpha);

    if (!success || pfd.historyLength > historyLength)
    {
        dLdHistoryLength = 0.0f;
    }

    // to do: history length and a bunch of other stuff
    PrevFrameData.Differential dpfd;

    dpfd.prevTemporalAccum = dLdPrevIllum.d;
    dpfd.prevMoments = dLdPrevMoments;
    dpfd.historyLength = dLdHistoryLength;

    //__bwd_diff(loadPrevData)(posH.xy, reprojectionParamters, kernelWeights, dpfd);


}

[BackwardDerivative(d_reproject)]
REPROJ_OUT reproject( FullScreenPassVsOut vsOut, float3 luminanceParams, float[4] reprojectionParamters, float[3] kernelWeights, float alpha, float momentsAlpha)
{
    const float4 posH = vsOut.posH;
    const int2 ipos = posH.xy;

    float3 illumination = demodulate(getNonemissiveColor(ipos), getAlbedo(ipos));

    // Workaround path tracer bugs. TODO: remove this when we can.
    if (isNaN(illumination.x) || isNaN(illumination.y) || isNaN(illumination.z))
    {
        illumination = float3(0, 0, 0);
    }

    var pfd = loadPrevData(posH.xy, reprojectionParamters, kernelWeights); // get temporal data with weights

    float2 prevMoments = pfd.prevMoments;
    float historyLength = pfd.historyLength;
    bool success = pfd.success;

    historyLength = min(reprojectionParamters[0], success ? historyLength + 1.0f : 1.0f);

    // it appears that the temporal accumulation does not have a constant alpha, instead it is adaptive

    // this adjusts the alpha for the case where insufficient history is available.
    // It boosts the temporal accumulation to give the samples equal weights in
    // the beginning.
    const float alphaAdjusted        = success ? max(alpha,        reprojectionParamters[1] / historyLength) : 1.0;
    const float alphaMomentsAdjusted = success ? max(momentsAlpha, reprojectionParamters[1] / historyLength) : 1.0;

    // compute first two moments of luminance
    float2 moments;
    moments.r = dot(illumination, luminanceParams.rgb);
    moments.g = moments.r * moments.r;

    float2 pm = moments;

    // temporal integration of the moments
    moments = lerp(prevMoments, moments, alphaMomentsAdjusted);

    float variance = max(0.f, moments.g - moments.r * moments.r);

    float4 initialEstimate = lerp(pfd.prevTemporalAccum, float4(illumination, 0), alphaAdjusted); ;
    float4 illum = initialEstimate;
    if (false && success)
    {
        illum = ff_mlp(initialEstimate, pfd.prevIllum, pfd.prevFiltered);
    }


    REPROJ_OUT psOut;
    // temporal integration of illumination
    psOut.OutIllumination = illum;
    // variance is propagated through the alpha channel
    psOut.OutIllumination.a = variance;
    psOut.OutMoments = moments;
    psOut.OutHistoryLength = historyLength;
    psOut.OutTemporalAccum = initialEstimate;

    return psOut;
}

/*
PS_OUT main(FullScreenPassVsOut vsOut) {
    DifferentialPair<float3> dpLuminanceParams = diffPair(dvLuminanceParams, float3(1.0, 0.0, 0.0));
    DifferentialPair<float[3]> dpReprojKernel = diffPair(dvReprojKernel, {0, 0, 0});
    DifferentialPair<float[4]> dpReprojParams = diffPair(dvReprojParams, {0, 0, 0, 0});
    DifferentialPair<float> dpAlpha = diffPair(dvAlpha, 0.0);
    DifferentialPair<float> dpMomentsAlpha = diffPair(dvMomentsAlpha, 0.0);

    float  finiteDiff = 1e-5;
    let nodiffres = reproject(vsOut, dpLuminanceParams.p - float3(finiteDiff, 0, 0), dpReprojParams.p, dpReprojKernel.p, dpAlpha.p, dpMomentsAlpha.p);
    let nodiffresStep = reproject(vsOut, dpLuminanceParams.p + float3(finiteDiff, 0, 0), dpReprojParams.p, dpReprojKernel.p, dpAlpha.p, dpMomentsAlpha.p);

    float finiteDiffDeriv = (nodiffresStep.OutMoments.x - nodiffres.OutMoments.x) / (2.0 * finiteDiff);

    REPROJ_OUT.Differential diffInput;

    diffInput.OutIllumination = float4(0.0);
    diffInput.OutMoments = float2(1.0, 0.0);
    diffInput.OutHistoryLength = 0.0;

    let result = __bwd_diff(reproject)(vsOut, dpLuminanceParams, dpReprojParams, dpReprojKernel, dpAlpha, dpMomentsAlpha, diffInput);

    float bwdDiffResult = dpLuminanceParams.d.x;

    PS_OUT ret;

    ret.OutMoments = nodiffres.OutMoments;
    ret.OutIllumination = nodiffres.OutIllumination;
    ret.OutHistoryLength = nodiffres.OutHistoryLength;

    ret.OutDebug = float4(dvReprojKernel[0], dvReprojKernel[1], dvReprojKernel[2], 0.0);

    return ret;
}
*/

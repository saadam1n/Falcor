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

cbuffer PerImageCB
{
    Texture2D gMotion;
    Texture2D gPositionNormalFwidth;
    Texture2D gColor;
    Texture2D gAlbedo;
    Texture2D gEmission;
    Texture2D gPrevIllum;
    Texture2D gPrevMoments;
    Texture2D gLinearZAndNormal;
    Texture2D gPrevLinearZAndNormal;
    Texture2D gPrevHistoryLength;

    float       dvAlpha;
    float       dvMomentsAlpha;

    float3 dvLuminanceParams;
    float dvReprojKernel[3];
    float dvReprojParams[4];

    float dvInputLayerWeights0[2];
    float dvInputLayerWeights1[2];
    float dvHiddenLayerWeights0[4];
    float dvHiddenLayerWeights1[4];
    float dvOutputLayerWeights[4];

    RWByteAddressBuffer daIllumination;
    RWByteAddressBuffer daMoments;
    RWByteAddressBuffer daHistoryLength;
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
    //storeDerivBuf4(daIllumination, p, grad, gAlbedo, getBilinearIndex(offset));
}

[BackwardDerivative(d_getPrevIllumBilinear)]
float4 getPrevIllumBilinear(int2 p, int2 offset)
{
    return gPrevIllum[p];
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
    //storeDerivBuf4(daIllumination, p, grad, gAlbedo, getBilateralIndex(offset));
}

[BackwardDerivative(d_getPrevIllumBilateral)]
float4 getPrevIllumBilateral(int2 p, int2 offset)
{
    return gPrevIllum[p];
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
    float4 prevIllum;
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

        float4 dLdPrevIllum = -grad.prevIllum / sumw;
        float2 dLdPrevMoments = -grad.prevMoments / sumw;

        for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++)
        {
            int2 loc = int2(posPrev) + offset[sampleIdx];
            if (v[sampleIdx])
            {
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

    pfd.prevIllum = float4(0, 0, 0, 0);
    pfd.prevMoments = float2(0, 0);
    pfd.success = false;

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
                pfd.prevIllum += w[sampleIdx] * getPrevIllumBilinear(loc, offset[sampleIdx]);
                pfd.prevMoments += w[sampleIdx] * getPrevMomentsBilinear(loc, offset[sampleIdx]);
                sumw += w[sampleIdx];
            }
        }

        // redistribute weights in case not all taps were used
        pfd.success = (sumw >= 0.01);
        pfd.prevIllum = pfd.prevIllum / sumw;
        pfd.prevMoments = pfd.prevMoments / sumw;
    }

    return pfd;
}

[BackwardDifferentiable]
PrevFrameData bilateralLoad(int2 iposPrev, float[3] kernelWeights, no_diff float2 depth, no_diff float3 normal, no_diff float normalFwidth, float[4] reprojectionParamters)
{
    float nValid = 0.0;
    float totalWeight = 0.0;

    PrevFrameData pfd;
    pfd.prevIllum = float4(0.0f);
    pfd.prevMoments = float2(0.0f);

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
                pfd.prevIllum += curWeight * getPrevIllumBilateral(p, int2(xx, yy));
                pfd.prevMoments += curWeight * getPrevMomentsBilateral(p, int2(xx, yy));
                nValid += 1.0;
                totalWeight += curWeight;
            }
        }
    }
    if (nValid > 0)
    {
        pfd.success = true;

        totalWeight = max(totalWeight, 1e-6);

        pfd.prevIllum /= totalWeight;
        pfd.prevMoments /= totalWeight;
    }

    return pfd;
}

// get the temporal data (this function does not accumulate)
[BackwardDifferentiable]
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
};

struct PS_OUT {
    float4 OutIllumination  : SV_TARGET0;
    float2 OutMoments       : SV_TARGET1;
    float  OutHistoryLength : SV_TARGET2;
    float4 OutDebug         : SV_TARGET3;
};

// perhaps instead of having a simple lerp function for TAA, we can have more complex curve
[BackwardDifferentiable]
REPROJ_OUT reproject( FullScreenPassVsOut vsOut, float3 luminanceParams, float[4] reprojectionParamters, float[3] kernelWeights, float alpha, float momentsAlpha,
    float[2] inputLayerWeights0, float[2] inputLayerWeights1, float[4] hiddenLayerWeights0, float[4] hiddenLayerWeights1, float[4] outputLayerWeights)
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

    float4 prevIllumination = pfd.prevIllum;
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

    float4 illum = float4(illumination, 0); // lerp(prevIllumination, float4(illumination, 0), alphaAdjusted)
    if (success)
    {
        float4 inputLayerOutputs[2];

        for (int i = 0; i < 2; i++)
        {
            inputLayerOutputs[i] = sigmoid(inputLayerWeights0[i] * float4(illumination, 0.0f) + inputLayerWeights1[i] * prevIllumination);
        }

        float4 hiddenLayerOutputs[4];
        for (int i = 0; i < 4; i++)
        {
            hiddenLayerOutputs[i] = sigmoid(hiddenLayerWeights0[i] * inputLayerOutputs[0] + hiddenLayerWeights1[i] * inputLayerOutputs[1]);
        }

        illum = float4(0.0f);
        for (int i = 0; i < 4; i++)
        {
            illum += outputLayerWeights[i] * hiddenLayerOutputs[i];
        }
        illum = softplus(illum);
    }


    REPROJ_OUT psOut;
    // temporal integration of illumination
    psOut.OutIllumination = illum;
    // variance is propagated through the alpha channel
    psOut.OutIllumination.a = variance;
    psOut.OutMoments = moments;
    psOut.OutHistoryLength = historyLength;

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

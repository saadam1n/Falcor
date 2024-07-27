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

void d_getPrevIllum(int2 ipos, float4 grad)
{
    // todo
}

[BackwardDerivative(d_getPrevIllum)]
float4 getPrevIllum(int2 ipos)
{
    return gPrevIllum[ipos];
}

void d_getPrevMoments(int2 ipos, float2 grad)
{
    // todo
}

[BackwardDerivative(d_getPrevMoments)]
float2 getPrevMoments(int2 ipos)
{
    return gPrevMoments[ipos].rg;
}

void d_getPrevHistoryLength(int2 ipos, float grad)
{
    // todo
}

[BackwardDerivative(d_getPrevHistoryLength)]
float getPrevHistoryLength(int2 ipos)
{
    return gPrevHistoryLength[ipos].r;
}

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

    float2 depth = getLinearZAndNormal(ipos).xy;
    float3 normal = oct_to_ndir_snorm(getLinearZAndNormal(ipos).zw);

    PrevFrameData pfd;

    pfd.prevIllum = float4(0, 0, 0, 0);
    pfd.prevMoments = float2(0, 0);

    bool v[4];
    const no_diff float2 posPrev = diffFloor(posH.xy) + motion.xy * imageDim;
    const int2 offset[4] = { int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1) };

    // check for all 4 taps of the bilinear filter for validity
    bool valid = false;
    for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++)
    {
        int2 loc = int2(posPrev) + offset[sampleIdx];
        float2 depthPrev = getPrevLinearZAndNormal(loc).xy;
        float3 normalPrev = oct_to_ndir_snorm(getPrevLinearZAndNormal(loc).zw);

        v[sampleIdx] = isReprjValid(iposPrev, depth.x, depthPrev.x, depth.y, normal, normalPrev, normalFwidth, reprojectionParamters);

        valid = valid || v[sampleIdx];
    }

    if (valid)
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
                pfd.prevIllum += w[sampleIdx] * getPrevIllum(loc);
                pfd.prevMoments += w[sampleIdx] * getPrevMoments(loc);
                sumw += w[sampleIdx];
            }
        }

        // redistribute weights in case not all taps were used
        valid = (sumw >= 0.01);
        pfd.prevIllum = valid ? pfd.prevIllum / sumw : float4(0, 0, 0, 0);
        pfd.prevMoments = valid ? pfd.prevMoments / sumw : float2(0, 0);
    }

    if (!valid) // perform cross-bilateral filter in the hope to find some suitable samples somewhere
    {
        float nValid = 0.0;
        float totalWeight = 0.0;

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
                    pfd.prevIllum += curWeight * getPrevIllum(p);
                    pfd.prevMoments += curWeight * getPrevMoments(p);
                    nValid += 1.0;
                    totalWeight += curWeight;
                }
            }
        }
        if (nValid > 0)
        {
            valid = true;

            totalWeight = max(totalWeight, 1e-6);

            pfd.prevIllum /= totalWeight;
            pfd.prevMoments /= totalWeight;
        }
    }

    if (valid)
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

    pfd.success = valid;

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
REPROJ_OUT reproject(FullScreenPassVsOut vsOut, float3 luminanceParams, float[4] reprojectionParamters, float[3] kernelWeights, float alpha, float momentsAlpha)
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
    const float alpha        = success ? max(alpha,        reprojectionParamters[1] / historyLength) : 1.0;
    const float alphaMoments = success ? max(momentsAlpha, reprojectionParamters[1] / historyLength) : 1.0;

    // compute first two moments of luminance
    float2 moments;
    moments.r = dot(illumination, luminanceParams.rgb);
    moments.g = moments.r * moments.r;

    float2 pm = moments;

    // temporal integration of the moments
    moments = lerp(prevMoments, moments, alphaMoments);

    float variance = max(0.f, moments.g - moments.r * moments.r);

    // variance *= computeVarianceScale(16, 16, alpha);

    REPROJ_OUT psOut;
    // temporal integration of illumination
    psOut.OutIllumination = lerp(prevIllumination, float4(illumination, 0), alpha);
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

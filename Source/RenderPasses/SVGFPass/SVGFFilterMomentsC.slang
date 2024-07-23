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

// slang doesn't support importing macros, so we have to redefine
#define NUM_WEIGHT_PARAMS 3
#define LUMINANCE_POLYN_DEGREE 3

cbuffer PerImageCB
{
    Texture2D gIllumination;
    Texture2D gMoments;
    Texture2D gHistoryLength;
    Texture2D gLinearZAndNormal;

    float3 dvLuminanceParams;
    float dvVarianceBoostFactor;

    float dvWeightFunctionParams[3];

    float dvSigmaL;
    float dvSigmaZ;
    float dvSigmaN;

    RWByteAddressBuffer   daIllumination;
    RWByteAddressBuffer   daHistoryLen;
    RWByteAddressBuffer   daMoments;

};




void d_getLinearZAndNormal(int2 ipos, float4 val) {}
[BackwardDerivative(d_getLinearZAndNormal)]
float4 getLinearZAndNormal(int2 ipos)
{
    return gLinearZAndNormal[ipos];
}

void d_getHistoryLength(int2 ipos, float val)
{
    storeDerivBuf4(daHistoryLen, ipos, float4(val, 0.0, 0.0, 0.0), gIllumination);
}
[BackwardDerivative(d_getHistoryLength)]
float getHistoryLength(int2 ipos)
{
    return gHistoryLength[ipos].x;
}

void d_getIllumination(int2 p, int2 offset, const bool ldcenter, float4 val)
{
    int storeIdx;
    if (ldcenter)
    {
        storeIdx = 49;
    }
    else
    {
        storeIdx = 7 * (offset.y + 3) + (offset.x + 3);
    }

    storeDerivBuf4(daIllumination, p, val, gIllumination, storeIdx);
}

[BackwardDerivative(d_getIllumination)]
float4 getIllumination(int2 p, int2 offset, const bool ldcenter)
{
    return gIllumination[p];
}


void d_getMoments(int2 p, int2 offset, float2 val)
{
    int storageIdx = 7 * (offset.y + 3) + (offset.x + 3);

    storeDerivBuf4(daMoments, p, float4(val, 0.0, 0.0), gIllumination, storageIdx);
}

[BackwardDerivative(d_getMoments)]
float2 getMoments(int2 p, int2 offset)
{
    return gMoments[p].xy;
}

uniform float SUM_W_ELIPSON = 1e-2f;

void d_filter_moments(
    FullScreenPassVsOut vsOut, inout DifferentialPair<float> historyParam, inout DifferentialPair<float3> luminanceParams,
    inout DifferentialPair<float[NUM_WEIGHT_PARAMS]> weightsParams, inout DifferentialPair<float> sigmaL, inout DifferentialPair<float> sigmaZ,
    inout DifferentialPair<float> sigmaN, float4 dLoss)
{
    float4 posH = vsOut.posH;
    int2 ipos = int2(posH.xy);

    float h = getHistoryLength(ipos).x;
    int2 screenSize = getTextureDims(gHistoryLength, 0);

    if (h < historyParam.p) // not enough temporal history available
    {
        float sumWIllumination = SUM_W_ELIPSON;
        float3 sumIllumination = float3(0.0, 0.0, 0.0);
        float2 sumMoments = float2(0.0, 0.0);

        const float4 illuminationCenter = getIllumination(ipos, int2(0, 0), true);
        const float lIlluminationCenter = dot(illuminationCenter.rgb, luminanceParams.p);

        const float2 zCenter = getLinearZAndNormal(ipos).xy;
        if (zCenter.x < 0)
        {
            // current pixel does not a valid depth => must be envmap => do nothing
            __bwd_diff(getIllumination)(ipos, int2(0, 0), true, dLoss);
            return;
        }
        const float3 nCenter = oct_to_ndir_snorm(getLinearZAndNormal(ipos).zw);
        const float phiLIllumination = sigmaL.p;
        const float phiDepth = sigmaZ.p * 3.0 * max(zCenter.y, 1e-8);

        // compute first and second moment spatially. This code also applies cross-bilateral
        // filtering on the input illumination.
        for (int yy = -3; yy <= 3; yy++)
        {
            for (int xx = -3; xx <= 3; xx++)
            {
                const int2 p = ipos + int2(xx, yy);
                const bool inside = all(p >= int2(0, 0)) && all(p < screenSize);
                const bool samePixel = (xx == 0 && yy == 0);
                const float kernel = 1.0;

                if (inside)
                {
                    const float3 illuminationP = getIllumination(p, int2(xx, yy), false).rgb;
                    const float2 momentsP = getMoments(p, int2(xx, yy)).xy;
                    const float lIlluminationP = dot(illuminationP.rgb, luminanceParams.p);
                    const float zP = getLinearZAndNormal(p).x;
                    const float3 nP = oct_to_ndir_snorm(getLinearZAndNormal(p).zw);

                    const float w = computeWeight(
                        zCenter.x, zP, phiDepth * length(float2(xx, yy)),
                        nCenter, nP, sigmaN.p,
                        lIlluminationCenter, lIlluminationP, phiLIllumination,
                        weightsParams.p);

                    sumWIllumination += w;
                    sumIllumination += illuminationP * w;
                    sumMoments += momentsP * w;
                }
            }
        }

        // Clamp sum to >0 to avoid NaNs.
        //sumWIllumination = max(sumWIllumination, SUM_W_ELIPSON);

        float3 normIllum = sumIllumination / sumWIllumination;
        float2 normMoments = sumMoments / sumWIllumination;

        // compute variance using the first and second moments
        float variance = normMoments.g - normMoments.r * normMoments.r;

        // give the variance a boost for the first frames
        float boostedVariance = variance * historyParam.p / h;

        float4 filtered = float4(normIllum, boostedVariance);

        float4 dLdFiltered = dLoss;

        float dLdHistoryParam = dLdFiltered.a * variance / h;
        historyParam = diffPair(historyParam.p, dLdHistoryParam);

        float dLdVariance = dLdFiltered.a * historyParam.p / h;
        float2 dLdNormMoments = dLdVariance * float2(1.0f, -2.0f * normMoments.r);

        float2 dLdSumMoments = dLdNormMoments / sumWIllumination;
        float3 dLdSumIllum = dLoss.rgb / sumWIllumination;

        float2 dLdSumWIllumFromMoments = -dLdNormMoments * sumMoments / (sumWIllumination * sumWIllumination);
        float3 dLdSumWIllumFromIllum = -dLoss.rgb * sumIllumination / (sumWIllumination * sumWIllumination);
        float dLdSumWIllum = dot(dLdSumWIllumFromMoments, 1.0f.xx) + dot(dLdSumWIllumFromIllum, 1.0f.xxx);

        // now reiterate through the loop, propagating derivatives
        DifferentialPair<float> dpzCenterX = diffPair(zCenter.x);
        float dLdZCenterX = float(0.0f);

        // we vary the input for this on on each iteration of the loop, so no diffpair
        float dLdPhiDepth = 0.0f;

        DifferentialPair<float3> dpnCenter = diffPair(nCenter);

        // diff pair in args
        float dLdSigmaN = 0.0f;

        DifferentialPair<float> dplIlluminationCenter = diffPair(lIlluminationCenter);
        float dLdlIlluminationCenter = 0.0f;

        DifferentialPair<float> dpPhiLIllumination = diffPair(phiLIllumination);
        float dLdPhiLIllumination = 0.0f;

        float dLdWeightParams[3];
        for (int i = 0; i < 3; i++)
        {
            dLdWeightParams[i] = 0.0f;
        }

        float3 dLdLuminanceParams = float3(0.0f);
        for (int yy = -3; yy <= 3; yy++)
        {
            for (int xx = -3; xx <= 3; xx++)
            {
                const int2 p = ipos + int2(xx, yy);
                const bool inside = all(p >= int2(0, 0)) && all(p < screenSize);
                const bool samePixel = (xx == 0 && yy == 0);
                const float kernel = 1.0;

                if (inside)
                {
                    const float3 illuminationP = getIllumination(p, int2(xx, yy), false).rgb;
                    const float2 momentsP = getMoments(p, int2(xx, yy)).xy;
                    const float lIlluminationP = dot(illuminationP.rgb, luminanceParams.p);
                    const float zP = getLinearZAndNormal(p).x;
                    const float3 nP = oct_to_ndir_snorm(getLinearZAndNormal(p).zw);

                    DifferentialPair<float> dpphiDepth = diffPair(phiDepth * length(float2(xx, yy)));
                    const float w = computeWeight(
                        zCenter.x, zP, dpphiDepth.p,
                        nCenter, nP, sigmaN.p,
                        lIlluminationCenter, lIlluminationP, phiLIllumination,
                        weightsParams.p);

                    // get moments in flight asap
                    float2 dLdMomentsP = dLdSumMoments * w;
                    __bwd_diff(getMoments)(p, int2(xx, yy), dLdMomentsP);


                    float dLdW = dLdSumWIllum + dot(dLdSumMoments * momentsP, 1.0f.xx) + dot(dLdSumIllum * illuminationP, 1.0f.xxx);

                    // now do the weight propagation
                    DifferentialPair<float> dpzP = diffPair(zP);
                    DifferentialPair<float3> dpnP = diffPair(nP);
                    DifferentialPair<float> dplIlluminationP = diffPair(lIlluminationP);

                    __bwd_diff(computeWeight )(dpzCenterX, dpzP,
                      dpphiDepth, dpnCenter, dpnP, sigmaN,
                      dplIlluminationCenter, dplIlluminationP,
                      dpPhiLIllumination, weightsParams, dLdW);

                    // get the accum operation in flight asap
                    // the illumination is modified via the final sum and weight
                    float3 dLdIlluminationP = dLdSumIllum * w + float3(dplIlluminationCenter.d * luminanceParams.p);
                    __bwd_diff(getIllumination)(p, int2(xx, yy), false, float4(dLdIlluminationP, 0.0f));

                    // propagate to the luminace params first
                    dLdLuminanceParams += dplIlluminationP.d * illuminationP.rgb;

                    // now that luminance is taken care of, let's work on the other paramters
                    dLdZCenterX += dpzCenterX.d;
                    dLdPhiDepth += dpphiDepth.d * length(float2(xx, yy));
                    dLdSigmaN += sigmaN.d;
                    dLdlIlluminationCenter += dplIlluminationCenter.d;
                    dLdPhiLIllumination += dpPhiLIllumination.d;

                    for (int i = 0; i < 3; i++)
                    {
                        dLdWeightParams[i] += weightsParams.d[i];
                    }
                }
            }
        }

        weightsParams = diffPair(weightsParams.p, dLdWeightParams);
        luminanceParams = diffPair(luminanceParams.p, dLdLuminanceParams + dLdlIlluminationCenter * illuminationCenter.rgb);

        float3 dLdIlluminationCenter = dLdSumIllum + dLdlIlluminationCenter * luminanceParams.p;
        __bwd_diff(getIllumination)(ipos, int2(0, 0), false, float4(dLdIlluminationCenter, 0.0f));

        // now we are done with that loop, let's back propagate everything
        sigmaN = diffPair(sigmaN.p, dLdSigmaN);
        sigmaZ = diffPair(sigmaZ.p, dLdPhiDepth * 3.0f * max(zCenter.y, 1e-8));
        sigmaL = diffPair(sigmaL.p, dLdPhiLIllumination);
    }
    else
    {
        // do nothing, pass data unmodified
        __bwd_diff(getIllumination)(ipos, int2(0, 0), true, dLoss);
        return;
    }
}

[BackwardDerivative(d_filter_moments)]
float4 filter_moments(FullScreenPassVsOut vsOut, float historyParam, float3 luminanceParams, float weightsParams[NUM_WEIGHT_PARAMS], float sigmaL, float sigmaZ, float sigmaN) {
    float4 posH = vsOut.posH;
    int2 ipos = int2(posH.xy);

    float h = getHistoryLength(ipos).x;
    int2 screenSize = getTextureDims(gHistoryLength, 0);

    if (h < historyParam) // not enough temporal history available
    {
        float sumWIllumination = SUM_W_ELIPSON; // add small bias to prevent naning (clamping doesn't seem to work)
        float3 sumIllumination = float3(0.0, 0.0, 0.0);
        float2 sumMoments = float2(0.0, 0.0);

        const float4 illuminationCenter = getIllumination(ipos, int2(0, 0), true);
        const float lIlluminationCenter = dot(illuminationCenter.rgb, luminanceParams);

        const float2 zCenter = getLinearZAndNormal(ipos).xy;
        if (zCenter.x < 0)
        {
            // current pixel does not a valid depth => must be envmap => do nothing
            return illuminationCenter;
        }
        const float3 nCenter = oct_to_ndir_snorm(getLinearZAndNormal(ipos).zw);
        const float phiLIllumination = sigmaL;
        const float phiDepth = sigmaZ * 3.0 * max(zCenter.y, 1e-8);

        // compute first and second moment spatially. This code also applies cross-bilateral
        // filtering on the input illumination.
        for (int yy = -3; yy <= 3; yy++)
        {
            for (int xx = -3; xx <= 3; xx++)
            {
                const int2 p = ipos + int2(xx, yy);
                const bool inside = all(p >= int2(0, 0)) && all(p < screenSize);
                const bool samePixel = (xx == 0 && yy == 0);
                const float kernel = 1.0;

                if (inside)
                {
                    const float3 illuminationP = getIllumination(p, int2(xx, yy), false).rgb;
                    const float2 momentsP = getMoments(p, int2(xx, yy)).xy;
                    const float lIlluminationP = dot(illuminationP.rgb, luminanceParams);
                    const float zP = getLinearZAndNormal(p).x;
                    const float3 nP = oct_to_ndir_snorm(getLinearZAndNormal(p).zw);

                    const float w = computeWeight(
                        zCenter.x, zP, phiDepth * length(float2(xx, yy)),
                        nCenter, nP, sigmaN,
                        lIlluminationCenter, lIlluminationP, phiLIllumination,
                        weightsParams);

                    sumWIllumination += w;
                    sumIllumination += illuminationP * w;
                    sumMoments += momentsP * w;
                }
            }
        }

        // Clamp sum to >0 to avoid NaNs.
        // sumWIllumination = max(sumWIllumination, SUM_W_ELIPSON);

        sumIllumination /= sumWIllumination;
        sumMoments /= sumWIllumination;

        // compute variance using the first and second moments
        float variance = sumMoments.g - sumMoments.r * sumMoments.r;

        // give the variance a boost for the first frames
        float boostedVariance = variance * historyParam / h;

        return float4(sumIllumination, boostedVariance);
    }
    else
    {
        // do nothing, pass data unmodified
        return getIllumination(ipos, int2(0, 0), true);
    }
}

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




void d_getLinearZAndNormal(int2 ipos, float4 val)
{
    // Something we really don't care about
}

void d_getHistoryLength(int2 ipos, float val)
{
    accumDerivBuf4(daHistoryLen, ipos, float4(val, 0.0, 0.0, 0.0), gIllumination);
}

void d_getIllumination(int2 ipos, int nsi, float4 val)
{
    accumDerivBuf4(daIllumination, ipos, val, gIllumination, nsi);
}

void d_getMoments(int2 ipos, int nsi, float2 val)
{
    accumDerivBuf4(daMoments, ipos, float4(val, 0.0, 0.0), gIllumination, nsi);
}



[BackwardDerivative(d_getLinearZAndNormal)]
float4 getLinearZAndNormal(int2 ipos)
{
    return gLinearZAndNormal[ipos];
}

[BackwardDerivative(d_getHistoryLength)]
float getHistoryLength(int2 ipos)
{
    return gHistoryLength[ipos].x;
}

[BackwardDerivative(d_getIllumination)]
float4 getIllumination(int2 ipos, int nsi)
{
    return gIllumination[ipos];
}

[BackwardDerivative(d_getMoments)]
float2 getMoments(int2 ipos, int nsi)
{
    return gMoments[ipos].xy;
}

[BackwardDifferentiable]
float4 filter_moments(FullScreenPassVsOut vsOut, float historyParam, float3 luminanceParams, float weightsParams[NUM_WEIGHT_PARAMS], float sigmaL, float sigmaZ, float sigmaN) {
    float4 posH = vsOut.posH;
    int2 ipos = int2(posH.xy);

    float h = getHistoryLength(ipos).x;
    int2 screenSize = getTextureDims(gHistoryLength, 0);

    int nsi_i = 0;
    int nsi_m = 0;

    if (h < historyParam) // not enough temporal history available
    {
        float sumWIllumination = 0.0;
        float3 sumIllumination = float3(0.0, 0.0, 0.0);
        float2 sumMoments = float2(0.0, 0.0);

        const float4 illuminationCenter = getIllumination(ipos, nsi_i++);
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

                int force_nsi_i = nsi_i++;
                int force_nsi_m = nsi_m++;

                if (inside)
                {
                    const float3 illuminationP = getIllumination(p, force_nsi_i).rgb;
                    const float2 momentsP = getMoments(p, force_nsi_m).xy;
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
        sumWIllumination = max(sumWIllumination, 1e-6f);

        sumIllumination /= sumWIllumination;
        sumMoments /= sumWIllumination;

        // compute variance using the first and second moments
        float variance = sumMoments.g - sumMoments.r * sumMoments.r;

        // give the variance a boost for the first frames
        variance *= historyParam / h;

        return float4(sumIllumination, variance.r);
    }
    else
    {
        // do nothing, pass data unmodified
        return getIllumination(ipos, nsi_i++);
    }
}
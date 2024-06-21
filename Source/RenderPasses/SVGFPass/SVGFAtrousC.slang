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

cbuffer PerImageCB
{
    Texture2D gAlbedo;
    Texture2D gHistoryLength;
    Texture2D gIllumination;
    Texture2D gLinearZAndNormal;

    int         gStepSize;

    float dvSigmaL;
    float dvSigmaZ;
    float dvSigmaN;

    float3 dvLuminanceParams;

    float dvWeightFunctionParams[3];
    float dvKernel[3];
    float dvVarianceKernel[2][2];

    RWByteAddressBuffer daIllumination;
};

void d_getLinearZAndNormal(int2 ipos, float4 val)
{
    // Something we really don't care about
}

[BackwardDerivative(d_getLinearZAndNormal)]
float4 getLinearZAndNormal(int2 ipos)
{
    return gLinearZAndNormal[ipos];
}

void d_getHistoryLength(int2 ipos, float val)
{
    // we will have to store it in a different way
}

[BackwardDerivative(d_getHistoryLength)]
float getHistoryLength(int2 ipos)
{
    return gHistoryLength[ipos].x;
}

void d_getIllumination(int2 ipos, int nsi, float4 val)
{
    accumDerivBuf4(daIllumination, ipos, val, gAlbedo, nsi);
}

[BackwardDerivative(d_getIllumination)]
float4 getIllumination(int2 ipos, int nsi)
{
    return gIllumination[ipos];
}

static int nsi = 0;
// computes a 3x3 gaussian blur of the variance, centered around
// the current pixel. nextStoreIndex will be 9 after this function
[BackwardDifferentiable]
float computeVarianceCenter(int2 ipos, float[2][2] kernel, int2 screenSize)
{
    float sum = 0.f;

    for (int yy = -1; yy <= 1; yy++)
    {
        for (int xx = -1; xx <= 1; xx++)
        {
            int2 p = ipos + int2(xx, yy);
            const float k = kernel[abs(xx)][abs(yy)];

            const bool inside = (all(p >= int2(0, 0)) && all(p < screenSize));
            if(inside)
                sum += getIllumination(p, nsi).a * k;
        }
    }

    return sum;
}

[BackwardDifferentiable]
float4 atrous(FullScreenPassVsOut vsOut, 
              float[3] kernelWeights, float [2][2] varianceKernel, float3 luminanceParams, float[3] weightsParams, 
              float sigmaL, float sigmaZ, float sigmaN) {
    const int2 ipos = int2(vsOut.posH.xy);
    const int2 screenSize = getTextureDims(gAlbedo, 0);

    const float epsVariance = 1e-10;

    // constant samplers to prevent the compiler from generating code which
    // fetches the sampler descriptor from memory for each texture access
    const float4 illuminationCenter = getIllumination(ipos, nsi);
    const float lIlluminationCenter = dot(illuminationCenter.rgb, luminanceParams);

    // variance, filtered using 3x3 gaussin blur
    const float var = computeVarianceCenter(ipos, varianceKernel, screenSize);

    const float2 zCenter = getLinearZAndNormal(ipos).xy;
    if (zCenter.x < 0)
    {
        // not a valid depth => must be envmap => do not filter
        return illuminationCenter;
    }
    const float3 nCenter = oct_to_ndir_snorm(getLinearZAndNormal(ipos).zw);

    const float phiLIllumination = sigmaL * sqrt(max(0.0, epsVariance + var.r));
    const float phiDepth = sigmaZ * max(zCenter.y, 1e-8) * gStepSize;

    // explicitly store/accumulate center pixel with weight 1 to prevent issues
    // with the edge-stopping functions
    float sumWIllumination = 1.0;
    float4 sumIllumination = illuminationCenter;

    for (int yy = -2; yy <= 2; yy++)
    {
        for (int xx = -2; xx <= 2; xx++)
        {
            const int2 p = ipos + int2(xx, yy) * gStepSize;
            const bool inside = all(p >= int2(0, 0)) && all(p < screenSize);

            const float kernel = kernelWeights[abs(xx)] * kernelWeights[abs(yy)];

            int force_nsi = nsi;

            if (inside && (xx != 0 || yy != 0)) // skip center pixel, it is already accumulated
            {
                const float4 illuminationP = getIllumination(p, force_nsi);
                const float lIlluminationP = dot(illuminationP.rgb, luminanceParams);
                const float zP = getLinearZAndNormal(p).x;
                const float3 nP = oct_to_ndir_snorm(getLinearZAndNormal(p).zw);

                // compute the edge-stopping functions
                const float w = computeWeight(
                    zCenter.x, zP, phiDepth * length(float2(xx, yy)),
                    nCenter, nP, sigmaN,
                    lIlluminationCenter, lIlluminationP, phiLIllumination,
                    weightsParams);

                const float wIllumination = w.x * kernel;

                // alpha channel contains the variance, therefore the weights need to be squared, see paper for the formula
                sumWIllumination += wIllumination;
                sumIllumination += float4(wIllumination.xxx, wIllumination * wIllumination) * illuminationP;
            }
        }
    }

    // renormalization is different for variance, check paper for the formula
    float4 filteredIllumination = float4(sumIllumination / float4(sumWIllumination.xxx, sumWIllumination * sumWIllumination));

    return filteredIllumination;
}


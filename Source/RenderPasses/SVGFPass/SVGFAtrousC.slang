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
    Texture2D gIllumination;
    Texture2D gLinearZAndNormal;

    int         gStepSize;

    float dvSigmaL[5][5];
    float dvSigmaZ[5][5];
    float dvSigmaN[5][5];

    float3 dvLuminanceParams;

    float dvWeightFunctionParams[3];
    float dvKernel[5][5];
    float dvVarianceKernel[2][2];

    RWByteAddressBuffer daSigmaL;
    RWByteAddressBuffer daSigmaZ;
    RWByteAddressBuffer daSigmaN;

    RWByteAddressBuffer daKernel;
    RWByteAddressBuffer daIllumination;
};

void d_getLinearZAndNormal(int2 ipos, const float4 val)
{
    // Something we really don't care about
}

[BackwardDerivative(d_getLinearZAndNormal)]
const float4 getLinearZAndNormal(int2 ipos)
{
    return gLinearZAndNormal[ipos];
}

void d_getIllumination(int2 p, int2 sampleIter, const bool varianceStage, const float4 val)
{
    int storageIdx;
    if (varianceStage)
    {
        storageIdx = 3 * (sampleIter.y + 1) + (sampleIter.x + 1);
    }
    else
    {
        // add 9 offset to prevent overwrites
        storageIdx = 5 * (sampleIter.y + 2) + (sampleIter.x + 2) + 9;
    }

    accumDerivBuf4(daIllumination, p, val, gAlbedo, storageIdx);
}

[BackwardDerivative(d_getIllumination), PreferRecompute]
const float4 getIllumination(int2 p, int2 sampleIter, const bool varianceStage)
{
    return gIllumination.Load(int3(p, 0));
}

void d_computerVarianceCenter(int2 screenSize, int2 ipos, inout DifferentialPair<float [2][2]> kernel, float dLoss)
{
    float dKernel[2][2];
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            dKernel[i][j] = 0.0f;
        }
    }

    float ksum = 0.0f;
    for (int yy = -1; yy <= 1; yy++)
    {
        for (int xx = -1; xx <= 1; xx++)
        {
            int2 p = ipos + int2(xx, yy);

            const bool inside = (all(p >= int2(0, 0)) && all(p < screenSize));
            if (inside)
            {
                ksum += kernel.p[abs(yy)][abs(xx)];
            }
        }
    }

    ksum = max(ksum, 1e-6f);
    float dLdvsum = dLoss / ksum;

    float vsum = 0.0f;
    for (int yy = -1; yy <= 1; yy++)
    {
        for (int xx = -1; xx <= 1; xx++)
        {
            int2 p = ipos + int2(xx, yy);

            const bool inside = (all(p >= int2(0, 0)) && all(p < screenSize));
            if (inside)
            {
                const float k = kernel.p[abs(yy)][abs(xx)];

                float variance = getIllumination(p, int2(xx, yy), true).a;
                vsum += variance * k;

                dKernel[abs(yy)][abs(xx)] += dLdvsum * variance;
                __bwd_diff(getIllumination)(p, int2(yy, xx), true, float4(0.0f, 0.0f, 0.0f, dLdvsum * k));
            }
        }
    }

    float dLdksum = dLoss * -vsum / (ksum * ksum);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            dKernel[i][j] += dLdksum;
        }
    }

    kernel = diffPair(kernel.p, dKernel);
}

// computes a 3x3 gaussian blur of the variance, centered around
// the current pixel. nextStoreIndex will be 9 after this function
[BackwardDerivative(d_computerVarianceCenter)]
float computeVarianceCenter(int2 screenSize, int2 ipos, float[2][2] kernel)
{
    float vsum = 0.f;
    float ksum = 0.0f;

    for (int yy = -1; yy <= 1; yy++)
    {
        for (int xx = -1; xx <= 1; xx++)
        {
            int2 p = ipos + int2(xx, yy);

            const bool inside = (all(p >= int2(0, 0)) && all(p < screenSize));
            if (inside)
            {
                const float k = kernel[abs(yy)][abs(xx)];

                vsum += getIllumination(p, int2(xx, yy), true).a * k;
                ksum += k;
            }
        }
    }

    vsum /= max(ksum, 1e-6f);

    return vsum;
}

const uniform float epsVariance = 1e-10;

void d_atrous( FullScreenPassVsOut vsOut, inout DifferentialPair<float [2][2]> varianceKernel,
    inout DifferentialPair<float3> luminanceParams, inout DifferentialPair<float[3]> weightsParams,
    float4 dLoss)
{
    const int2 ipos = int2(vsOut.posH.xy);
    const int2 screenSize = getTextureDims(gAlbedo, 0);

    // constant samplers to prevent the compiler from generating code which
    // fetches the sampler descriptor from memory for each texture access
    const float4 illuminationCenter = getIllumination(ipos, int2(0, 0), false);
    const float lIlluminationCenter = dot(illuminationCenter.rgb, luminanceParams.p);

    const float2 zCenter = getLinearZAndNormal(ipos).xy;
    if (zCenter.x < 0)
    {
        // not a valid depth => must be envmap => do not filter
        __bwd_diff(getIllumination)(ipos, int2(0, 0), false, dLoss);
        return;
    }
    // variance, filtered using 3x3 gaussin blur
    const float var = computeVarianceCenter(screenSize, ipos, varianceKernel.p);

    const float3 nCenter = oct_to_ndir_snorm(getLinearZAndNormal(ipos).zw);

    const float phiLIlluminationBase = sqrt(max(0.0, epsVariance + var.r));
    const float phiDepthBase = max(zCenter.y, 1e-8) * gStepSize;
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

            int yri = yy + 2;
            int xri = xx + 2;

            const float kernel = dvKernel[yri][xri];

            if (inside && (xx != 0 || yy != 0)) // skip center pixel, it is already accumulated
            {
                const float4 illuminationP = getIllumination(p, int2(xx, yy), false);
                const float lIlluminationP = dot(illuminationP.rgb, luminanceParams.p);
                const float zP = getLinearZAndNormal(p).x;
                const float3 nP = oct_to_ndir_snorm(getLinearZAndNormal(p).zw);

                // compute the edge-stopping functions
                const float w = computeWeight(
                    zCenter.x, zP, dvSigmaZ[yri][xri] * phiDepthBase * length(float2(xx, yy)),
                    nCenter, nP, dvSigmaN[yri][xri],
                    lIlluminationCenter, lIlluminationP, dvSigmaL[yri][xri] * phiLIlluminationBase,
                    weightsParams.p);

                const float wIllumination = w.x * kernel;

                // alpha channel contains the variance, therefore the weights need to be squared, see paper for the formula
                sumWIllumination += wIllumination;
                sumIllumination += float4(wIllumination.xxx, wIllumination * wIllumination) * illuminationP;
            }
        }
    }

    // renormalization is different for variance, check paper for the formula
    // const float4 filteredIllumination = float4(sumIllumination / float4(sumWIllumination.xxx, sumWIllumination * sumWIllumination));

    float4 dLdSumIllum = dLoss / float4(sumWIllumination.xxx, sumWIllumination * sumWIllumination);
    float4 dLdSumW4 = -dLoss * sumIllumination / float4((sumWIllumination * sumWIllumination).xxx, 0.5f * sumWIllumination * sumWIllumination * sumWIllumination);

    #ifdef FILTER_NAN_PROTECT
    for (int i = 0; i < 4; i++)
    {
        if (isnan(sumIllumination[i]))
        {
            dLdSumIllum[i] = 0.0f;
            dLdSumW4[i] = 0.0f;
        }
    }
    #endif

    float dLdSumW = dot(dLdSumW4, 1.0f.xxxx);

    // go though the loop again, propagating derivatives
    DifferentialPair<float> dpzCenterX = diffPair(zCenter.x);
    float dLdZCenterX = float(0.0f);

    DifferentialPair<float3> dpnCenter = diffPair(nCenter);

    DifferentialPair<float> dplIlluminationCenter = diffPair(lIlluminationCenter);
    float dLdlIlluminationCenter = 0.0f;

    float dLdPhiLIlluminationBase = 0.0f;

    float dLdWeightParams[3];
    for (int i = 0; i < 3; i++)
    {
        dLdWeightParams[i] = 0.0f;
    }

    float3 dLdLuminanceParams = float3(0.0f);

    float4 writeValKernel;
    float4 writeValSigmaL;
    float4 writeValSigmaZ;
    float4 writeValSigmaN;

    int sampleIdx = 0;

    for (int yy = -2; yy <= 2; yy++)
    {
        for (int xx = -2; xx <= 2; xx++)
        {
            const int2 p = ipos + int2(xx, yy) * gStepSize;
            const bool inside = all(p >= int2(0, 0)) && all(p < screenSize);

            int yri = yy + 2;
            int xri = xx + 2;
            int writeIdx = 5 * yri + xri;

            const float kernel = dvKernel[yri][xri];

            if (inside && (xx != 0 || yy != 0)) // skip center pixel, it is already accumulated
            {
                const float4 illuminationP = getIllumination(p, int2(xx, yy), false);
                const float lIlluminationP = dot(illuminationP.rgb, luminanceParams.p);
                const float zP = getLinearZAndNormal(p).x;
                const float3 nP = oct_to_ndir_snorm(getLinearZAndNormal(p).zw);

                // compute the edge-stopping functions
                DifferentialPair<float> dpphiDepth = diffPair(dvSigmaZ[yri][xri] * phiDepthBase * length(float2(xx, yy)));
                const float w = computeWeight(
                    zCenter.x, zP, dpphiDepth.p,
                    nCenter, nP, dvSigmaN[yri][xri],
                    lIlluminationCenter, lIlluminationP, dvSigmaL[yri][xri] * phiLIlluminationBase,
                    weightsParams.p);

                const float wIllumination = w.x * kernel;

                // find both sources of loss for wIllum
                float dLdwIllum = dLdSumW + dot(dLdSumIllum * float4(illuminationP.rgb, 2.0f * wIllumination * illuminationP.a), 1.0f.xxxx);
                float dLdWeight = dLdwIllum * kernel;

                // propagate to the kernels
                writeValKernel[writeIdx % 4] = dLdwIllum * w;

                // now do the weight propagation
                DifferentialPair<float> dpzP = diffPair(zP);
                DifferentialPair<float3> dpnP = diffPair(nP);
                DifferentialPair<float> dplIlluminationP = diffPair(lIlluminationP);

                DifferentialPair<float> dpPhiLIllumination = diffPair(dvSigmaL[yri][xri] * phiLIlluminationBase);
                DifferentialPair<float> dpSigmaN = diffPair(dvSigmaN[yri][xri]);
                __bwd_diff(computeWeight )(dpzCenterX, dpzP,
                  dpphiDepth, dpnCenter, dpnP, dpSigmaN,
                    dplIlluminationCenter, dplIlluminationP,
                    dpPhiLIllumination, weightsParams, dLdWeight);

                // get the accum operation in flight asap
                // the illumination is modified via the final sum and weight
                float4 dLdIlluminationP = dLdSumIllum * float4(wIllumination.xxx, wIllumination * wIllumination) + float4(dplIlluminationCenter.d * luminanceParams.p, 0.0f);
                __bwd_diff(getIllumination)(p, int2(xx, yy), false, dLdIlluminationP);

                // propagate to the luminace params first
                dLdLuminanceParams += dplIlluminationP.d * illuminationP.rgb;

                // now that luminance is taken care of, let's work on the other paramters
                dLdZCenterX += dpzCenterX.d;
                writeValSigmaZ[writeIdx % 4] = dpphiDepth.d * length(float2(xx, yy)) * phiDepthBase;
                writeValSigmaN[writeIdx % 4] = dpSigmaN.d;
                writeValSigmaL[writeIdx % 4] = dpPhiLIllumination.d * phiLIlluminationBase;

                dLdlIlluminationCenter += dplIlluminationCenter.d;
                dLdPhiLIlluminationBase += dpPhiLIllumination.d * dvSigmaL[yri][xri];
                for (int i = 0; i < 3; i++)
                {
                    dLdWeightParams[i] += weightsParams.d[i];
                }

            }
            else
            {
                writeValKernel[writeIdx % 4] = 0.0f;
                writeValSigmaL[writeIdx % 4] = 0.0f;
                writeValSigmaZ[writeIdx % 4] = 0.0f;
                writeValSigmaN[writeIdx % 4] = 0.0f;
            }
            // write if full
            if (writeIdx % 4 == 3 || writeIdx == 24)
            {
                storeDerivBuf4(daKernel, ipos, writeValKernel, gAlbedo, writeIdx / 4);
                storeDerivBuf4(daSigmaL, ipos, writeValSigmaL, gAlbedo, writeIdx / 4);
                storeDerivBuf4(daSigmaZ, ipos, writeValSigmaZ, gAlbedo, writeIdx / 4);
                storeDerivBuf4(daSigmaN, ipos, writeValSigmaN, gAlbedo, writeIdx / 4);
            }
        }
    }

    weightsParams = diffPair(weightsParams.p, dLdWeightParams);
    luminanceParams = diffPair(luminanceParams.p, dLdLuminanceParams + dLdlIlluminationCenter * illuminationCenter.rgb);

    float4 dLdIlluminationCenter = dLdSumIllum + float4(dLdlIlluminationCenter * luminanceParams.p, 0.0f);
    __bwd_diff(getIllumination)(ipos, int2(0, 0), false, dLdIlluminationCenter);


    // now we are done with that loop, let's back propagate everything

    float dLdVariance = dLdPhiLIlluminationBase * 0.5f / sqrt(max(1e-5f, epsVariance + var.r));
    __bwd_diff(computeVarianceCenter)(screenSize, ipos, varianceKernel, dLdVariance);
}

[BackwardDerivative(d_atrous)]
float4 atrous(FullScreenPassVsOut vsOut, float [2][2] varianceKernel,
              float3 luminanceParams, float[3] weightsParams) {
    const int2 ipos = int2(vsOut.posH.xy);
    const int2 screenSize = getTextureDims(gAlbedo, 0);


    // constant samplers to prevent the compiler from generating code which
    // fetches the sampler descriptor from memory for each texture access
    const float4 illuminationCenter = getIllumination(ipos, int2(0, 0), false);
    const float lIlluminationCenter = dot(illuminationCenter.rgb, luminanceParams);

    const float2 zCenter = getLinearZAndNormal(ipos).xy;
    if (zCenter.x < 0)
    {
        // not a valid depth => must be envmap => do not filter
        return illuminationCenter;
    }
    // variance, filtered using 3x3 gaussin blur
    const float var = computeVarianceCenter(screenSize, ipos, varianceKernel);

    const float3 nCenter = oct_to_ndir_snorm(getLinearZAndNormal(ipos).zw);

    const float phiLIlluminationBase = sqrt(max(0.0, epsVariance + var.r));
    const float phiDepthBase = max(zCenter.y, 1e-8) * gStepSize;

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

            int yri = yy + 2;
            int xri = xx + 2;

            const float kernel = dvKernel[yri][xri];

            if (inside && (xx != 0 || yy != 0)) // skip center pixel, it is already accumulated
            {
                const float4 illuminationP = getIllumination(p, int2(xx, yy), false);
                const float lIlluminationP = dot(illuminationP.rgb, luminanceParams);
                const float zP = getLinearZAndNormal(p).x;
                const float3 nP = oct_to_ndir_snorm(getLinearZAndNormal(p).zw);

                // compute the edge-stopping functions
                const float w = computeWeight(
                    zCenter.x, zP, dvSigmaZ[yri][xri] * phiDepthBase * length(float2(xx, yy)),
                    nCenter, nP, dvSigmaN[yri][xri],
                    lIlluminationCenter, lIlluminationP, dvSigmaL[yri][xri] * phiLIlluminationBase,
                    weightsParams );

                const float wIllumination = w.x * kernel;
                // alpha channel contains the variance, therefore the weights need to be squared, see paper for the formula
                sumWIllumination += wIllumination;
                sumIllumination += float4(wIllumination.xxx, wIllumination * wIllumination) * illuminationP;
            }
        }
    }

    #ifdef FILTER_NAN_PROTECT
    for (int i = 0; i < 4; i++)
    {
        if (isnan(sumIllumination[i]))
        {
            sumIllumination[i] = 0.0f;
        }
    }
    #endif

    // renormalization is different for variance, check paper for the formula
    const float4 filteredIllumination = float4(sumIllumination / float4(sumWIllumination.xxx, sumWIllumination * sumWIllumination));
    return filteredIllumination;
}


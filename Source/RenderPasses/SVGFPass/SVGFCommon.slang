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
import Utils.Color.ColorHelpers;

const int2 getTextureDims(Texture2D tex, uint mip)
{
    return int2(1920, 1080);
}

[BackwardDifferentiable]
float computeWeight(
    float depthCenter, float depthP, float phiDepth,
    float3 normalCenter, float3 normalP, float phiNormal,
    float luminanceIllumCenter, float luminanceIllumP, float phiIllum,
    float[3] weightParams)
{
    const float weightNormal  = weightParams[0] * pow(saturate(dot(normalCenter, normalP)), phiNormal);
    const float weightZ       = (phiDepth == 0) ? 0.0f : abs(depthCenter - depthP) / phiDepth;
    const float weightLillum  = weightParams[1] * abs(luminanceIllumCenter - luminanceIllumP) / phiIllum;

    float weightIllum   = exp(-weightParams[2] * (max(weightLillum, 0.0) + max(weightZ, 0.0))) * weightNormal;

    // nan protection, I am not sure if this is the source of nans
    if (isnan(weightIllum))
    {
        weightIllum = 0.0f; 
    }

    return weightIllum;
}

struct FullScreenPassVsOut
{
    float2 texC : TEXCOORD;
#ifndef _VIEWPORT_MASK
    float4 posH : SV_POSITION;
#else
    float4 posH : POSITION;
#endif
};

[BackwardDifferentiable]
float getAlbedoLum(float3 albedo)
{
    float lum = dot(albedo.rgb, float3(1.0));
    return lum;
}

[BackwardDifferentiable]
float3 demodulate(float3 c, float3 albedo)
{
    float lum = getAlbedoLum(albedo);
    if(lum > 0.01)
        return c / max(albedo, float3(0.001, 0.001, 0.001));
    else    
        return c;
}

[BackwardDifferentiable]
float4 remodulate(float4 c, float4 albedo) {

    float lum = getAlbedoLum(albedo.rgb);
    if(lum > 0.01)
        return c * max(albedo, float4(0.001, 0.001, 0.001, 1.0)); // don't touch the variance
    else    
        return c;
}

float4 computeDebugDiffError(float finite, float bwd) {
    float MIN_DERIV = 0.025;

    float val;
    if(abs(bwd) < MIN_DERIV || abs(bwd) < MIN_DERIV) {
        val = 0.0;
    } else {
        val = abs((bwd - finite) / finite);
    }

    return float4(val);
}

static const float QUANTIZATION = 65536.0 * 8.0;

int3 quantVal3(float3 val) {
    return int3(QUANTIZATION * val);
}

int4 quantVal4(float3 val) {
    return int4(int3(QUANTIZATION * val), 0);
}

int4 quantVal4(float4 val) {
    return int4(QUANTIZATION * val);
}

float3 dequant3(int3 val) {
    return float3(val) / QUANTIZATION;
}

float4 dequant4(int4 val) {
    return float4(val) / QUANTIZATION;
}

uint getBufIdx3(int2 ipos, Texture2D tex) {
    uint2 dim = getTextureDims(tex, 0);
    uint base_addr = sizeof(int3) * (dim.x * ipos.y + ipos.x);
    return base_addr;
}

const uint getBufIdx4(int2 ipos, Texture2D tex, int idx) {
    const uint2 dim = getTextureDims(tex, 0);
    const uint base_addr = ((dim.x * ipos.y + ipos.x) + dim.x * dim.y * idx) * sizeof(int4);
    return base_addr;
}

//#define ATOMIC_STORE

// tex is a texture with equal dimensions to rwaddrbuff
void accumDerivBuf4(RWByteAddressBuffer rwaddrbuff, int2 ipos, float4 val, Texture2D tex, int idx = 0, int offset = 0) {
    const uint base_addr = getBufIdx4(ipos, tex, idx) + offset;

    //int4 a = quantVal4(val);
    const int4 a = reinterpret<int4>(val);
    
    #ifdef ATOMIC_STORE
    for(int i = 0; i < 4; i++) {
        uint addr = base_addr + 4 * i;
        int dummy;
        rwaddrbuff.InterlockedAdd(addr, a[i], dummy);
    }
    #else
    int status;
    rwaddrbuff.Store4(base_addr, a);
    #endif
}

const float4 readDerivBuf4(RWByteAddressBuffer rwaddrbuff, int2 ipos, Texture2D tex, int idx = 0, int offset = 0) {
    const uint base_addr = getBufIdx4(ipos, tex, idx) + offset;

    uint status;
    const uint4 temp = rwaddrbuff.Load4(base_addr, status);

    //return dequant4(int4(temp));
    return reinterpret<float4>(temp);
}











void accumDerivBuf3(RWByteAddressBuffer rwaddrbuff, int2 ipos, float3 val, Texture2D tex) {
    uint base_addr = getBufIdx3(ipos, tex);

    #ifdef ATOMIC_STORE
    int3 a = quantVal3(val);
    for(int i = 0; i < 3; i++) {
        uint addr = base_addr + 4 * i;
        int dummy;
        rwaddrbuff.InterlockedAdd(addr, a[i], dummy);
    }
    #else
    // idk figure out what to put here later
    //rwaddrbuff.Store3(base_addr, a, status);
    #endif
}

// TODO: figure out how to properly read from a rwbyteaddressbuf

float3 readDerivBuf3(RWByteAddressBuffer rwaddrbuff, int2 ipos, Texture2D tex) {
    uint base_addr = getBufIdx3(ipos, tex);

    int3 temp;
    for(int i = 0; i < 3; i++) {
        uint addr = base_addr + 4 * i;
        rwaddrbuff.InterlockedAdd(addr, 0, temp[i]);
    }

    return dequant3(temp);
}
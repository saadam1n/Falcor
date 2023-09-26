import SVGFCommon;

cbuffer PerImageCB
{
    Texture2D   gAlbedo;
    Texture2D   gEmission;
    Texture2D   gIllumination;

    RWByteAddressBuffer daIllumination;
};

// Read gColor
void d_getIllum(int2 ipos, float3 val)
{
    // this we actually care about, as this feeds into the previous stage's output 
    uint2 dim = getTextureDims(gIllumination, 0);
    uint base_addr = sizeof(int4) * (dim.x * ipos.y + ipos.x);

    int4 a = quantVal4(val);
    for(int i = 0; i < 3; i++) {
        uint addr = base_addr + sizeof(int) * i;
        int dummy;
        daIllumination.InterlockedAdd(addr, a[i], dummy);
    }
}

// Read gAlbedo
void d_getAlbedo(int2 ipos, float3 val)
{

}

// Read gEmission
void d_getEmission(int2 ipos, float3 val)
{

}

[BackwardDerivative(d_getIllum)]
float3 getIllum(int2 ipos)
{
    return gIllumination[ipos].rgb;
}

[BackwardDerivative(d_getAlbedo)]
float3 getAlbedo(int2 ipos)
{
    return gAlbedo[ipos].rgb;
}

[BackwardDerivative(d_getEmission)]
float3 getEmission(int2 ipos)
{
    return gEmission[ipos].rgb;
}

[BackwardDifferentiable]
float4 final_mod(FullScreenPassVsOut vsOut) {
    const int2 ipos = int2(vsOut.posH.xy);
    return float4(remodulate(getIllum(ipos), getAlbedo(ipos)) + getEmission(ipos), 0.0);
}

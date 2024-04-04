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
    // add one float of padding because that is what the atrous pipeline uses
    // accumDerivBuf4(daIllumination, ipos, float4(val, 0.0), gIllumination);
}

// Read gAlbedo
void d_getAlbedo(int2 ipos, float3 val)
{
    // constant that we do not care about
}

// Read gEmission
void d_getEmission(int2 ipos, float3 val)
{
    // constant that we do not care about
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

import SVGFCommon;

cbuffer PerImageCB
{
    Texture2D   gAlbedo;
    Texture2D   gEmission;
    Texture2D   gIllumination;

    RWByteAddressBuffer daIllumination;
};

// Read gColor
void d_getIllum(int2 ipos, float4 val)
{
    // this we actually care about, as this feeds into the previous stage's output
    // add one float of padding because that is what the atrous pipeline uses
    accumDerivBuf4(daIllumination, ipos, val, gIllumination);
}

// Read gAlbedo
void d_getAlbedo(int2 ipos, float4 val)
{
    // constant that we do not care about
}

// Read gEmission
void d_getEmission(int2 ipos, float4 val)
{
    // constant that we do not care about
}

[BackwardDerivative(d_getIllum)]
float4 getIllum(int2 ipos)
{
    return gIllumination[ipos];
}

[BackwardDerivative(d_getAlbedo)]
float4 getAlbedo(int2 ipos)
{
    return gAlbedo[ipos];
}

[BackwardDerivative(d_getEmission)]
float4 getEmission(int2 ipos)
{
    return gEmission[ipos];
}

[BackwardDifferentiable]
float4 final_mod(FullScreenPassVsOut vsOut) {
    const int2 ipos = int2(vsOut.posH.xy);

    return remodulate(getIllum(ipos), getAlbedo(ipos)) + getEmission(ipos);
}

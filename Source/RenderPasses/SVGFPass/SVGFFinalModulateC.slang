import SVGFCommon;

cbuffer PerImageCB
{
    Texture2D   gAlbedo;
    Texture2D   gEmission;
    Texture2D   gIllumination;
};

// Read gColor
void d_getIllum(int2 ipos, float3 val)
{

}

// Read gAlbedo
void d_getAlbedo(int2 ipos, float3 val)
{

}

// Read gEmission
void d_getEmission(int2 ipos, float3 val)
{

}

[BackwardDerivative(d_getColor)]
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
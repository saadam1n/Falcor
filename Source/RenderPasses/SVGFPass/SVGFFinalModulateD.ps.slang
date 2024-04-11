import SVGFCommon;
import SVGFFinalModulateC;

cbuffer PerImageCB_D
{
    RWByteAddressBuffer drFilteredImage;
};

void main(FullScreenPassVsOut vsOut) : SV_TARGET0
{
    const int2 ipos = int2(vsOut.posH.xy);

    __bwd_diff(final_mod)(vsOut,  float4(1.0f));

    //float3 test_write_val = float3(0.5f);
    //accumDerivBuf4(daIllumination, ipos, float4(test_write_val, 0.0), gIllumination);
    //daIllumination.Store4(getBufIdx4(ipos, gIllumination), quantVal4(test_write_val));
}
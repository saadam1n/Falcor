import SVGFCommon;

cbuffer PerImageCB_Combine
{
    // this will come from the final modulate pass
    RWByteAddressBuffer dLdA2; 
    // this will be from the second atrous pass
    RWByteAddressBuffer dA2dA1i;
    // this will be from the first atrous pass
    RWByteAddressBuffer dA1dk1;

    // we will write this somewhere
    RWByteAddressBuffer dLdk1;

    int gStepSize;
    Texture2D gAlbedo;
};

void main(FullScreenPassVsOut vsOut)
{
    // for this pixel, let's accumulate everything
    const int2 ipos = int2(vsOut.posH.xy);

    // go through each pixel in the kernel
    // dL/dk1 = dL/dA2 * sum of each dA2/dA1,i * dA1,i/dk1
    float3 dIncomingLoss = readDerivBuf4(dLdA2, ipos, gAlbedo).rgb;
    float3 color_dLdk1 = float3(0.0f);

    int pixelID = 0;
    for(int yy = -2; yy <= 2; yy++) 
    {
        for(int xx = -2; xx <= 2; xx++)
        {
            int2 p1 = ipos + int2(0, 1080 * pixelID++);
            int2 p2 = ipos + int2(xx, yy) * gStepSize;

            float3 interPassDeriv = readDerivBuf4(dA2dA1i, p1, gAlbedo).rgb;
            float a1k1Deriv = readDerivBuf4(dA1dk1, p2, gAlbedo).x;

            color_dLdk1 += dIncomingLoss * interPassDeriv * a1k1Deriv;
        }
    }

    float combined_dLdk1 = dot(color_dLdk1, float3(1.0f));

    accumDerivBuf4(dLdk1, ipos, float4(combined_dLdk1), gAlbedo);

}

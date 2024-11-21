import Common;

#define KERNEL_DIM 13

cbuffer PerImageCB
{
    Texture2D src;
    CbufferFloatArray<KERNEL_DIM, KERNEL_DIM> kernel;
};

float4 main(FullScreenPassVsOut vsOut) : SV_TARGET0
{
    const int2 ipos = int2(vsOut.posH.xy);

    float4 sum = 0.0f;
    for (int i = 0; i < KERNEL_DIM; i++) {
        for (int j = 0; j < KERNEL_DIM; j++) {
            int2 p = ipos + int2(i, j) - KERNEL_DIM / 2;
            sum += src[p] * kernel[i, j]; // this is gonna look weird
        }
    }

    return sum;
}

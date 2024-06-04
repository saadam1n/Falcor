import SVGFAtrousC;
import SVGFAtrousDCommon;
import SVGFCommon;

void main(FullScreenPassVsOut vsOut)
{
    const int2 ipos = int2(vsOut.posH.xy);
    float4 dIllum = readDerivBuf4(drIllumination, ipos, gAlbedo);

    var pdv = calcAtrousDiffPairs(vsOut, int2(0, 0), dIllum);

    storeAtrousPdv(ipos, pdv);
}

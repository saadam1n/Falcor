import SVGFAtrousC;
import SVGFCommon;

struct FS_OUT {
    float4 filter_result : SV_TARGET0;
    float4 debug_buf : SV_TARGET1;
};

FS_OUT main(FullScreenPassVsOut vsOut)
{
    const int2 ipos = int2(vsOut.posH.xy);

    FS_OUT fs_out;

    fs_out.filter_result = atrous(vsOut, dvKernel, dvVarianceKernel, dvLuminanceParams, dvWeightFunctionParams, dvSigmaL, dvSigmaZ, dvSigmaN);

    return fs_out;
}

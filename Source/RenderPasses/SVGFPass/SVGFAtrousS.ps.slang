import SVGFAtrousC;
import SVGFCommon;

struct FS_OUT {
    float4 filter_result : SV_TARGET0;
    float4 debug_buf : SV_TARGET1;
};

FS_OUT main(FullScreenPassVsOut vsOut)
{
    FS_OUT fs_out;

    fs_out.filter_result = atrous(vsOut, dvVarianceKernel, dvLuminanceParams, dvWeightFunctionParams);

    return fs_out;
}

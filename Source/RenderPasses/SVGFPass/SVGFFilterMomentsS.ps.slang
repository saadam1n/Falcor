import SVGFCommon;
import SVGFFilterMomentsC;

struct FS_OUT {
    float4 illum_and_var : SV_TARGET0;
    float4 debug_buf : SV_TARGET1;
};

// compute moments
FS_OUT main(FullScreenPassVsOut vsOut)
{
    FS_OUT fs_out;

    fs_out.illum_and_var = filter_moments(vsOut, dvVarianceBoostFactor, dvLuminanceParams, dvWeightFunctionParams, dvSigmaL, dvSigmaZ, dvSigmaN);
    fs_out.debug_buf = 0.0;

    return fs_out;
}

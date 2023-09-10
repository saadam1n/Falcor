import SVGFReprojectC;
import SVGFCommon;

PS_OUT main(FullScreenPassVsOut vsOut) {
    let nodiffres = reproject(vsOut, dvLuminanceParams, dvReprojParams, dvReprojKernel, dvAlpha, dvMomentsAlpha);

    PS_OUT ret;

    ret.OutMoments = nodiffres.OutMoments;
    ret.OutIllumination = nodiffres.OutIllumination;
    ret.OutHistoryLength = nodiffres.OutHistoryLength;

    return ret;
}
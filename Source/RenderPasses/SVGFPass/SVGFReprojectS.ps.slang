import SVGFReprojectC;
import SVGFCommon;

PS_OUT main(FullScreenPassVsOut vsOut) {
    let nodiffres = reproject(vsOut, dvLuminanceParams, dvReprojParams, dvReprojKernel, dvAlpha, dvMomentsAlpha,
        dvInputLayerWeights0, dvInputLayerWeights1, dvHiddenLayerWeights0, dvHiddenLayerWeights1, dvOutputLayerWeights);

    PS_OUT ret;

    ret.OutMoments = nodiffres.OutMoments;
    ret.OutIllumination = nodiffres.OutIllumination;
    ret.OutHistoryLength = nodiffres.OutHistoryLength;

    return ret;
}

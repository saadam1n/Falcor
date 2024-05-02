add_plugin(SVGFReference)

target_sources(SVGFReference PRIVATE
    SVGFAtrous.ps.slang
    SVGFCommon.slang
    SVGFFilterMoments.ps.slang
    SVGFFinalModulate.ps.slang
    SVGFPackLinearZAndNormal.ps.slang
    SVGFReference.cpp
    SVGFReference.h
    SVGFReproject.ps.slang
)

target_copy_shaders(SVGFReference RenderPasses/SVGFReference)

target_source_group(SVGFReference "RenderPasses")

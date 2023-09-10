add_plugin(SVGFPass)

target_sources(SVGFPass PRIVATE
    SVGFAtrousC.slang
    SVGFAtrousD.ps.slang
    SVGFAtrousS.ps.slang

    SVGFCommon.slang

    SVGFFilterMomentsC.slang
    SVGFFilterMomentsD.ps.slang
    SVGFFilterMomentsS.ps.slang

    SVGFFinalModulateC.slang
    SVGFFinalModulateD.ps.slang
    SVGFFinalModulateS.ps.slang

    SVGFPackLinearZAndNormal.ps.slang
    SVGFPass.cpp
    SVGFPass.h

    SVGFReprojectC.slang
    SVGFReprojectD.ps.slang
    SVGFReprojectS.ps.slang
)

target_copy_shaders(SVGFPass RenderPasses/SVGFPass)

target_source_group(SVGFPass "RenderPasses")

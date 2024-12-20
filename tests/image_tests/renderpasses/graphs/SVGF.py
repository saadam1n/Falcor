from falcor import *

def render_graph_SVGF():
    g = RenderGraph("SVGF")
    SVGFPass = createPass("SVGFPass", {'Enabled': True,  'FeedbackTap': 1, 'VarianceEpsilon': 9.999999747378752e-05, 'PhiColor': 10.0, 'PhiNormal': 128.0, 'Alpha': 0.05000000074505806, 'MomentsAlpha': 0.20000000298023224})
    g.addPass(SVGFPass, "SVGFPass")
    GBufferRaster = createPass("GBufferRaster", {'cull': 'Back'})
    g.addPass(GBufferRaster, "GBufferRaster")
    PathTracer = createPass("PathTracer")
    g.addPass(PathTracer, "PathTracer")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("PathTracer.color", "SVGFPass.Color")
    g.addEdge("PathTracer.albedo", "SVGFPass.Albedo")
    g.addEdge("GBufferRaster.vbuffer", "PathTracer.vbuffer")
    g.addEdge("GBufferRaster.emissive", "SVGFPass.Emission")
    g.addEdge("GBufferRaster.posW", "SVGFPass.WorldPosition")
    g.addEdge("GBufferRaster.guideNormalW", "SVGFPass.WorldNormal")
    g.addEdge("GBufferRaster.pnFwidth", "SVGFPass.PositionNormalFwidth")
    g.addEdge("GBufferRaster.linearZ", "SVGFPass.LinearZ")
    g.addEdge("GBufferRaster.mvec", "SVGFPass.MotionVec")

    g.addEdge("SVGFPass.Filtered image", "ToneMapper.src")

    # Mark this first so it appears in the list first
    #g.markOutput("ToneMapper.dst")
    g.markOutput("SVGFPass.Filtered image")
    g.markOutput("SVGFPass.Reference")
    g.markOutput("SVGFPass.Loss")
    g.markOutput("SVGFPass.CenterLoss")
    g.markOutput("SVGFPass.GradientLoss")
    g.markOutput("SVGFPass.TemporalLoss")
    g.markOutput("ToneMapper.dst")



    g.markOutput("SVGFPass.DerivVerify")
    g.markOutput("SVGFPass.FuncLower")
    g.markOutput("SVGFPass.FuncUpper")
    g.markOutput("SVGFPass.FdCol")
    g.markOutput("SVGFPass.BdCol")


    g.markOutput("ToneMapper.dst")
    g.markOutput("SVGFPass.DebugBuf")
    g.markOutput("SVGFPass.DerivVerify")
    g.markOutput("PathTracer.color")
    g.markOutput("PathTracer.albedo")
    g.markOutput("GBufferRaster.posW")
    g.markOutput("GBufferRaster.guideNormalW")
    g.markOutput("GBufferRaster.emissive")
    g.markOutput("GBufferRaster.pnFwidth")
    g.markOutput("GBufferRaster.linearZ")
    g.markOutput("GBufferRaster.mvec")
    return g

SVGF = render_graph_SVGF()
try: m.addGraph(SVGF)
except NameError: None

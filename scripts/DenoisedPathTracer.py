from falcor import *

def render_graph_PathTracer():
    g = RenderGraph("PathTracer")

    VBufferRT = createPass("VBufferRT", {'samplePattern': SamplePattern.Stratified, 'sampleCount': 16, 'useAlphaTest': True})
    g.addPass(VBufferRT, "VBufferRT")

    PathTracer = createPass("PathTracer", {'samplesPerPixel': 1})
    g.addPass(PathTracer, "PathTracer")

    # Since we are using SVGF which does reprojection instead of accumulation, I'll disable this
    #AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': AccumulatePrecision.Single})
    #g.addPass(AccumulatePass, "AccumulatePass")

    # I'm unaware of the parameters for this, hopefully the SVGF pass can default to some sensible set of values
    SVGFPass = createPass("SVGFPass")
    g.addPass(SVGFPass, "SVGFPass")

    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("VBufferRT.mvec", "PathTracer.mvec")

    # Comment this since we have disabled the accumulation pass
    #g.addEdge("PathTracer.color", "AccumulatePass.input")
    #g.addEdge("AccumulatePass.output", "ToneMapper.src")

    # Instaed of accumulation, let's do SVGF
    g.addEdge("PathTracer.color", "SVGFPass.Color")
    g.addEdge("PathTracer.albedo", "SVGFPass.Albedo")
    g.addEdge("PathTracer.nrdEmission", "SVGFPass.Emission")
    g.addEdge("PathTracer.reflectionPosW", "SVGFPass.WorldPosition")
    g.addEdge("VBufferRT.mvec", "SVGFPass.MotionVec")
    g.addEdge("VBufferRT.depth", "SVGFPass.LinearZ")
    # Maybe this works?
    g.addEdge("PathTracer.guideNormal", "SVGFPass.WorldNormal")
    # I have no idea what this parameter is
    g.addEdge("PathTracer.color", "SVGFPass.PositionNormalFwidth")

    g.addEdge("SVGFPass.Filtered image", "ToneMapper.src")

    g.markOutput("ToneMapper.dst")
    return g

PathTracer = render_graph_PathTracer()
try: m.addGraph(PathTracer)
except NameError: None

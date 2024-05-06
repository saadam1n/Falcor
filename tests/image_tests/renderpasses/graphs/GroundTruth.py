from falcor import *

def render_graph_GroundTruth():
    g = RenderGraph("GroundTruth")
    PathTracer = createPass("PathTracer", {'samplesPerPixel': 1})
    g.addPass(PathTracer, "PathTracer")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Center', 'sampleCount': 16, 'useAlphaTest': True})
    g.addPass(VBufferRT, "VBufferRT")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    DatasetSaver = createPass("DatasetSaver")
    g.addPass(DatasetSaver, "DatasetSaver")


    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")

    g.addEdge("PathTracer.color", "DatasetSaver.realtime");
    g.addEdge("AccumulatePass.output", "DatasetSaver.reference");

    # Final frame output
    g.markOutput("ToneMapper.dst")
    g.markOutput("ToneMapper.dst", TextureChannelFlags.Alpha)

    # Path tracer outputs
    g.markOutput("PathTracer.color")
    g.markOutput("PathTracer.albedo")
    g.markOutput("PathTracer.specularAlbedo")
    g.markOutput("PathTracer.indirectAlbedo")
    g.markOutput("PathTracer.guideNormal")
    g.markOutput("PathTracer.reflectionPosW")
    g.markOutput("PathTracer.rayCount")
    g.markOutput("PathTracer.pathLength")


    g.markOutput("DatasetSaver.dummyOut");

    return g

PathTracer = render_graph_GroundTruth()
try: m.addGraph(PathTracer)
except NameError: None

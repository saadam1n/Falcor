from falcor import *

def render_graph_GroundTruth():
    g = RenderGraph("GroundTruth")


    GBufferRaster = createPass("GBufferRaster", {'cull': 'Back'})
    g.addPass(GBufferRaster, "GBufferRaster")
    PathTracer = createPass("PathTracer", {'samplesPerPixel': 1})
    g.addPass(PathTracer, "PathTracer")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    DatasetSaver = createPass("DatasetSaver")
    g.addPass(DatasetSaver, "DatasetSaver")

    # PT inputs
    g.addEdge("GBufferRaster.vbuffer", "PathTracer.vbuffer")

    # Map PT output to accumulate pass
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")

    # These are the PT paths that would normally go to SVGF
    g.addEdge("PathTracer.color", "DatasetSaver.Color");
    g.addEdge("PathTracer.albedo", "DatasetSaver.Albedo")

    # These are the GBuffer passes that would go to SVGF
    g.addEdge("GBufferRaster.emissive", "DatasetSaver.Emission")
    g.addEdge("GBufferRaster.posW", "DatasetSaver.WorldPosition")
    g.addEdge("GBufferRaster.guideNormalW", "DatasetSaver.WorldNormal")
    g.addEdge("GBufferRaster.pnFwidth", "DatasetSaver.PositionNormalFwidth")
    g.addEdge("GBufferRaster.linearZ", "DatasetSaver.LinearZ")
    g.addEdge("GBufferRaster.mvec", "DatasetSaver.MotionVec")

    # Add reference output to our dataset saver pass as well
    g.addEdge("AccumulatePass.output", "DatasetSaver.Reference");


    # Final frame output, use this while debugging
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

    # This is just to trick Falcor into executing this dummy pass
    g.markOutput("DatasetSaver.dummyOut");

    return g

PathTracer = render_graph_GroundTruth()
try: m.addGraph(PathTracer)
except NameError: None

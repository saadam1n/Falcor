add_plugin(DatasetSaver)

target_sources(DatasetSaver PRIVATE
    DatasetSaver.cpp
    DatasetSaver.h
)

target_source_group(DatasetSaver "RenderPasses")

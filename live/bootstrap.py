from dataclasses import dataclass

from live_io.frame_source import WebcamFrameSource
from live_io.output_sink import DebugTextOutputSink
from live_io.query_input import StaticQueryInput
from live_runtime import DepthWorker, GroundingWorker, load_depth_model, load_model


@dataclass(slots=True)
class RuntimeDependencies:
    predictor: object
    llm: object
    grounding_worker: GroundingWorker
    depth_worker: DepthWorker | None
    frame_source: WebcamFrameSource
    query_input: StaticQueryInput
    output_sink: DebugTextOutputSink


def build_runtime_dependencies(config):
    grounding_processor, grounding_model, predictor, llm = load_model(config.model)

    if config.enable_depth:
        depth_model, depth_checkpoint_path = load_depth_model(
            depth_encoder=config.depth_encoder,
            depth_dataset=config.depth_dataset,
            depth_max_depth=config.depth_max_depth,
            depth_checkpoint=config.depth_checkpoint,
        )
        print(f"depth checkpoint {depth_checkpoint_path}")
        depth_worker = DepthWorker(
            depth_model,
            input_size=config.depth_input_size,
        ).start()
    else:
        depth_worker = None

    frame_source = WebcamFrameSource(config.camera_index).start()
    query_input = StaticQueryInput(config.query)
    output_sink = DebugTextOutputSink()
    grounding_worker = GroundingWorker(
        grounding_processor,
        grounding_model,
        box_threshold=config.box_threshold,
        text_threshold=config.text_threshold,
    ).start()

    return RuntimeDependencies(
        predictor=predictor,
        llm=llm,
        grounding_worker=grounding_worker,
        depth_worker=depth_worker,
        frame_source=frame_source,
        query_input=query_input,
        output_sink=output_sink,
    )


def release_runtime_dependencies(deps: RuntimeDependencies):
    if deps.depth_worker is not None:
        deps.depth_worker.release()
    deps.grounding_worker.release()
    deps.frame_source.release()

from dataclasses import dataclass

from live_io.audio_input import AudioCaptureConfig, MicQueryInput
from live_io.frame_source import FrameSource, ServerStreamFrameSource, WebcamFrameSource
from live_io.output_sink import DebugTextOutputSink
from live_io.query_input import QueryInput, StaticQueryInput
from live_runtime import DepthWorker, GroundingWorker, load_depth_model, load_model


@dataclass(slots=True)
class RuntimeDependencies:
    predictor: object
    llm: object
    grounding_worker: GroundingWorker
    depth_worker: DepthWorker | None
    frame_source: FrameSource
    query_input: QueryInput
    output_sink: DebugTextOutputSink


def build_frame_source(config) -> FrameSource:
    if config.frame_source == "webcam":
        return WebcamFrameSource(config.camera_index).start()
    if config.frame_source == "server":
        return ServerStreamFrameSource(
            config.stream_url,
            poll_interval_s=config.stream_poll_interval_s,
            timeout_s=config.stream_timeout_s,
        ).start()
    raise ValueError(f"Unsupported frame source: {config.frame_source}")


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

    frame_source = build_frame_source(config)
    if config.query_input == "audio":
        if not hasattr(llm, "model") or not hasattr(llm.model, "transcribe_audio"):
            raise ValueError(
                "Audio query input requires an OpenAI-backed model with transcription support."
            )
        audio_config = AudioCaptureConfig(
            wake_phrase=config.wake_phrase,
            sample_rate=config.audio_sample_rate,
            channels=1,
            chunk_size=config.audio_chunk_size,
            wake_window_s=config.wake_window_s,
            wake_cooldown_s=config.wake_cooldown_s,
            command_max_record_s=config.command_max_record_s,
            silence_threshold=config.audio_silence_threshold,
            min_silence_duration_s=config.min_silence_duration_s,
            min_command_duration_s=config.min_command_duration_s,
            pre_roll_s=config.audio_pre_roll_s,
            transcription_model=config.transcription_model,
            language=config.transcription_language,
            input_device_index=config.audio_input_device_index,
        )
        query_input = MicQueryInput(
            audio_config,
            transcriber=llm.model.transcribe_audio,
        )
    else:
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
    deps.query_input.release()

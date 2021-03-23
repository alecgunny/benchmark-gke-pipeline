import argparse
import typing

import numpy as np
import tritonclient.grpc as triton
from stillwater import (
    DummyDataGenerator,
    MultiSourceDataGenerator,
    ThreadedMultiStreamInferenceClient
)


def main(
    url: str,
    model_name: str,
    model_version: int,
    num_clients: int,
    sequence_id: int,
    generation_rate: float,
    num_iterations: int = 10000,
    warm_up: typing.Optional[int] = None,
    file_prefix: typing.Optional[str] = None,
    latency_threshold: float = 0.1,
    queue_threshold_us: float = 20000
):
    client = ThreadedMultiStreamInferenceClient(
        url=url,
        model_name=model_name,
        model_version=model_version,
        name="client"
    )

    output_pipes = {}
    for i in range(num_clients):
        seq_id = sequence_id + i

        sources = []
        for state_name, shape in client.states:
            sources.append(DummyDataGenerator(
                shape=shape,
                name=state_name,
                generation_rate=generation_rate
            ))
        source = MultiSourceDataGenerator(sources)
        pipe = client.add_data_source(source, seq_id)
        output_pipes[seq_id] = pipe

    warm_up_client = triton.InferenceServerClient(url)
    warm_up_inputs = []
    for input in client.model_config.input:
        x = triton.InferInput(input.name, tuple(input.dims), input.datatype)
        x.set_data_from_numpy(np.random.randn(*input.dims).astype("float32"))
        warm_up_inputs.append(x)

    for i in range(warm_up):
        warm_up_client.infer(model_name, warm_up_inputs, str(model_version))

    file_prefix = "" if file_prefix is None else (file_prefix + "_")
    monitor = client.monitor(
        output_pipes,
        server_monitor={
            "output_file": f"{file_prefix}server-stats.csv",
            "snapshotter_queue": queue_threshold_us
        },
        client_monitor={
            "output_file": f"{file_prefix}client-stats.csv",
            "latency": latency_threshold
        },
        timeout=10
    )

    print(f"Gathering performance metrics over {num_iterations} iterations")
    packages_recvd = 0
    for seq_id, _ in monitor:
        if seq_id is not None:
            packages_recvd += 1

        if packages_recvd == num_iterations:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    client_parser = parser.add_argument_group(
        title="Client",
        description=(
            "Arguments for instantiation the Triton "
            "client instance"
        )
    )
    client_parser.add_argument(
        "--url",
        type=str,
        default="localhost:8001",
        help="Server URL"
    )
    client_parser.add_argument(
        "--model-name",
        type=str,
        default="gwe2e",
        help="Name of model to send requests to"
    )
    client_parser.add_argument(
        "--model-version",
        type=int,
        default=1,
        help="Model version to send requests to"
    )
    client_parser.add_argument(
        "--sequence-id",
        type=int,
        default=1001,
        help="Sequence identifier to use for the client stream"
    )

    data_parser = parser.add_argument_group(
        title="Data",
        description="Arguments for instantiating the client data sources"
    )
    data_parser.add_argument(
        "--generation-rate",
        type=float,
        required=True,
        help="Rate at which to generate data"
    )

    runtime_parser = parser.add_argument_group(
        title="Run Options",
        description="Arguments parameterizing client run"
    )
    runtime_parser.add_argument(
        "--num-iterations",
        type=int,
        default=10000,
        help="Number of requests to get for profiling"
    )
    runtime_parser.add_argument(
        "--num-clients",
        type=int,
        default=1,
        help="Number of clients to run simultaneously"
    )
    runtime_parser.add_argument(
        "--warm-up",
        type=int,
        default=None,
        help="Number of warm up requests to make"
    )
    runtime_parser.add_argument(
        "--file_prefix",
        type=str,
        default=None,
        help="Prefix to attach to monitor files"
    )
    flags = parser.parse_args()
    main(**vars(flags))

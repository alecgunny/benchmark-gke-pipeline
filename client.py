import argparse
import time
import typing

import numpy as np
import tritonclient.grpc as triton
from stillwater import StreamingInferenceClient, DummyDataGenerator

from measurement import ServerStatsMonitor, ClientStatsMonitor
from client_utils import log, Pipeline


def main(
    url: str,
    model_name: str,
    model_version: int,
    num_clients: int,
    sequence_id: int,
    generation_rate: float,
    num_iterations: int = 10000,
    warm_up: typing.Optional[int] = None,
    file_prefix: typing.Optional[str] = None
):
    clients, data_generators, output_pipes = [], [], {}
    for i in range(num_clients):
        seq_id = sequence_id + i
        client = StreamingInferenceClient(
            url=url,
            model_name=model_name,
            model_version=model_version,
            name=f"client_{seq_id}",
            sequence_id=sequence_id
        )
        clients.append(client)

        if i == 0 and warm_up is not None:
            warm_up_client = triton.InferenceServerClient(url)
            for input in client._inputs.values():
                input.set_data_from_numpy(
                    np.random.randn(*input.shape()).astype("float32")
                )
            for i in range(warm_up):
                _ = warm_up_client.infer(
                    model_name,
                    list(client._inputs.values()),
                    str(model_version)
                )

        for input_name, input in client.inputs.items():
            data_gen = DummyDataGenerator(
                input.shape()[1:],
                f"{input_name}_{seq_id}",
                generation_rate=generation_rate
            )
            client.add_parent(data_gen, input_name=input_name)
            data_generators.append(data_gen)

        for output in client.outputs:
            name = "{}_{}".format(output.name(), seq_id)
            output_pipes[name] = client.add_child(output.name())

    file_prefix = "" if file_prefix is None else (file_prefix + "_")
    client_stats_monitor = ClientStatsMonitor(
        f"{file_prefix}client-stats.csv", clients
    )
    server_stats_monitor = ServerStatsMonitor(
        f"{file_prefix}server-stats.csv",
        url,
        model_name,
        monitor="snapshotter",
        limit=10**6
    )

    with Pipeline(clients + data_generators, output_pipes) as pipeline:
        client_stats_monitor.start()
        server_stats_monitor.start()

        packages_recvd = 0

        log.info(
            f"Gathering performance metrics over {num_iterations} iterations"
        )
        last_package_time = time.time()
        msg = ""
        max_msg_length = len(msg)
        latency, throughput, request_rate = 0, 0, 0
        while packages_recvd < num_iterations:
            error = None
            for i in range(num_clients):
                pipes = {}
                for output in client.outputs:
                    name = "{}_{}".format(output.name(), sequence_id + i)
                    pipes[name] = output_pipes[name]

                try:
                    package = pipeline.get(pipes, timeout=5)
                except RuntimeError as e:
                    error = e
                    break

                if package is not None:
                    last_package_time = time.time()
                    packages_recvd += 1
                elif time.time() - last_package_time > 20:
                    error = "timeout"
                    break

            # check if we had any issues
            if error is not None and str(error).startswith("Server"):
                print("\n")
                msg = str(e).split(": ", maxsplit=1)[1]
                log.error(f"Encountered server error {msg}")
                log.error(f"Breaking after {packages_recvd} steps")
                break
            elif error == "timeout":
                print("\n")
                error = "timeout"
                log.error(
                    f"Timed out, breaking after {packages_recvd} steps"
                )
                break
            elif error is not None:
                print("\n")
                raise RuntimeError(str(error))

            # check to make sure our monitors are going ok
            client_monitor_error = client_stats_monitor.error
            if client_monitor_error is not None:
                print("\n")
                log.error("Encountered client monitor error {}".format(
                    str(client_monitor_error))
                )
                server_stats_monitor.stop()
                break

            server_monitor_error = server_stats_monitor.error
            if server_monitor_error is not None:
                if isinstance(server_monitor_error, Exception):
                    print("\n")
                    log.error(
                        "Encountered server monitor error {}".format(
                            str(server_monitor_error)
                        )
                    )
                    client_stats_monitor.stop()
                    break
                else:
                    print("\n")
                    log.error(
                        "Snapshotter queue length reached {} ms".format(
                            server_monitor_error / 10**6
                        )
                    )

            # update some of our metrics
            latency = client_stats_monitor.latency or latency
            throughput = client_stats_monitor.throughput or throughput
            request_rate = client_stats_monitor.request_rate or request_rate

            msg = f"Average latency: {latency} us, "
            msg += f"Average throughput: {throughput:0.1f} frames / s, "
            msg += f"Average request rate: {request_rate:0.1f} frames / s"

            max_msg_length = max(max_msg_length, len(msg))
            msg += " " * (max_msg_length - len(msg))
            print(msg, end="\r", flush=True)

    print("\n")
    log.info(msg)

    client_stats_monitor.stop()
    client_stats_monitor.join(0.5)

    server_stats_monitor.stop()
    server_stats_monitor.join(0.5)


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
        "--file-prefix",
        type=str,
        default=None,
        help="Prefix to attach to monitor files"
    )
    flags = parser.parse_args()
    main(**vars(flags))

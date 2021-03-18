import argparse
import time

import numpy as np
import tritonclient.grpc as triton
from stillwater import StreamingInferenceClient, DummyDataGenerator

from measurement import ServerStatsMonitor, ClientStatsMonitor
from client_utils import log, Pipeline


def main(
    url: str,
    model_name: str,
    model_version: int,
    sequence_id: int,
    generation_rate: float,
    num_iterations: int = 10000
):
    client = StreamingInferenceClient(
        url=url,
        model_name=model_name,
        model_version=model_version,
        name="client",
        sequence_id=sequence_id
    )

    # do a quick warm up
    warm_up_client = triton.InferenceServerClient(url)
    for input in client._inputs.values():
        input.set_data_from_numpy(
            np.random.randn(*input.shape()).astype("float32")
        )

    log.info("Warming up with 10 requests")
    for i in range(10):
        _ = warm_up_client.infer(
            model_name,
            list(client._inputs.values()),
            str(model_version)
        )

    processes = [client]
    for input_name, input in client.inputs.items():
        data_gen = DummyDataGenerator(
            input.shape()[1:], input_name, generation_rate=generation_rate
        )
        client.add_parent(data_gen)
        processes.append(data_gen)

    out_pipes = {}
    for output in client.outputs:
        out_pipes[output.name()] = client.add_child(output.name())

    client_stats_monitor = ClientStatsMonitor("client.csv", client._metric_q)
    server_stats_monitor = ServerStatsMonitor(
        "server.csv", url, model_name, monitor="snapshotter", limit=10**6
    )

    with Pipeline(processes, out_pipes) as pipeline:
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
            try:
                package = pipeline.get(timeout=5)
            except RuntimeError as e:
                if str(e).startswith("Server"):
                    print("\n")
                    msg = str(e).split(": ", maxsplit=1)[1]
                    log.error(f"Encountered server error {msg}")
                    log.error(f"Breaking after {packages_recvd} steps")
                    break

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

            if package is None:
                if time.time() - last_package_time > 20:
                    print("\n")
                    log.error(
                        f"Timed out, breaking after {packages_recvd} steps"
                    )
                    break
                continue
            last_package_time = time.time()
            packages_recvd += 1

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
    flags = parser.parse_args()
    main(**vars(flags))

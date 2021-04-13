import argparse
import logging
import os
import time
import typing
from queue import Empty

import numpy as np
import tritonclient.grpc as triton
from stillwater import (
    DummyDataGenerator,
    MultiSourceGenerator,
    ThreadedMultiStreamInferenceClient
)
from stillwater.utils import ExceptionWrapper


def _normalize_file_prefix(file_prefix):
    if file_prefix is None:
        file_prefix = ""
    elif not os.path.isdir(file_prefix):
        file_prefix = file_prefix + "_"
    return file_prefix


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
    latency_threshold: float = 1.,
    queue_threshold_us: float = 100000
):
    client = ThreadedMultiStreamInferenceClient(
        url=url,
        model_name=model_name,
        model_version=model_version,
        qps_limit=generation_rate,
        name="client"
    )

    output_pipes = {}
    for i in range(num_clients):
        seq_id = sequence_id + i

        sources = []
        for state_name, shape in client.states.items():
            sources.append(DummyDataGenerator(
                shape=shape,
                name=state_name,
            ))
        source = MultiSourceGenerator(sources)
        pipe = client.add_data_source(source, str(seq_id), seq_id)
        output_pipes[seq_id] = pipe

    warm_up_client = triton.InferenceServerClient(url)
    warm_up_inputs = []
    for input in client.model_metadata.inputs:
        x = triton.InferInput(input.name, input.shape, input.datatype)
        x.set_data_from_numpy(np.random.randn(*input.shape).astype("float32"))
        warm_up_inputs.append(x)

    for i in range(warm_up):
        warm_up_client.infer(model_name, warm_up_inputs, str(model_version))

    file_prefix = _normalize_file_prefix(file_prefix)
    logging.info(
        f"Gathering performance metrics over {num_iterations} iterations"
    )

    num_packages_received = 0
    bars = "|" + " " * 25 + "|"
    max_msg = f" {num_iterations}/{num_iterations}"
    max_len = len(bars) + len(max_msg)

    client.start()
    try:
        while True:
            for seq_id, pipe in output_pipes.items():
                if not pipe.poll():
                    continue
                x = pipe.recv()
                if isinstance(x, ExceptionWrapper):
                    x.reraise()
                num_packages_received += 1
                if num_packages_received == num_iterations:
                    break

            num_equal_signs = num_packages_received * 25 // num_iterations
            num_spaces = 25 - num_equal_signs
            msg = "|" + "=" * num_equal_signs + " " * num_spaces + "|"
            msg += f" {num_packages_received}/{num_iterations}"
            num_spaces = " " * (max_len - len(msg))
            print(msg + num_spaces, end="\r", flush=True)
    finally:
        client.stop()
        client.join(1)
        try:
            client.close()
        except ValueError:
            client.terminate()
            time.sleep(0.1)
            client.close()
            logging.warning("Client closed ungracefully")

    with open(f"{file_prefix}client-stats.csv", "w") as f:
        columns = [
            "sequence_id",
            "message_start",
            "request_send",
            "request_get",
            "request_return",
        ]
        f.write(",".join(columns))
        while True:
            try:
                measurements = client._metric_q.get_nowait()
            except Empty:
                break
            if measurements[0] == "start_time":
                start_time = measurements[1]
                continue

            measurements = list(measurements)
            sequence_id = measurements.pop(0)
            measurements = [i - start_time for i in measurements]
            measurements = [sequence_id] + measurements
            f.write(",".join(map(str, measurements)))


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
    runtime_parser.add_argument(
        "--queue-threshold-us",
        type=float,
        default=100000,
        help="Maximum allowable queuing time in microseconds"
    )
    runtime_parser.add_argument(
        "--latency-threshold",
        type=float,
        default=1.,
        help="Maximum allowable end-to-end latency in seconds"
    )
    runtime_parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file to write to"
    )
    runtime_parser.add_argument(
        "--num-retries",
        type=int,
        default=0,
        help="Retry attempts if running into thread issue"
    )
    flags = vars(parser.parse_args())

    log_file = flags.pop("log_file")
    if log_file is not None:
        logging.basicConfig(filename=log_file, level=logging.INFO)
    else:
        import sys
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    with open("/proc/cpuinfo", "r") as f:
        cpuinfo = f.read().split("\n")
    families = [i.split(": ")[1] for i in cpuinfo if i.startswith("cpu family")]
    models = [i.split(": ")[1] for i in cpuinfo if i.startswith("model\t")]
    for f, m in zip(families, models):
        logging.info(f"CPU family {f}, model {m}")

    num_violations = 0
    num_retries = flags.pop("num_retries")
    while num_violations <= num_retries:
        try:
            main(**flags)
            break
        # except MonitoredMetricViolationException as e:
        #     logging.exception("Metric violation")
        #     if "latency" not in str(e):
        #         raise
        #     else:
        #         file_prefix = _normalize_file_prefix(flags["file_prefix"])
        #         df = pd.read_csv(f"{file_prefix}server-stats.csv")

        #         df["step"] = df.index // 6
        #         elapsed = df.groupby("step")[["interval"]].agg("mean").cumsum()
        #         elapsed.columns = ["elapsed"]
        #         df = df.join(elapsed, on="step")
        #         df = df[df.elapsed < df.elapsed.max() - 2]

        #         if np.percentile(df.queue, 99) < 200:
        #             num_violations += 1
        #             logging.warning("Queue times stable, retrying")
        #             continue
        #         else:
        #             raise
        except Exception:
            logging.exception("Fatal error")
            raise
    else:
        logging.exception("Too many violations")
        raise RuntimeError

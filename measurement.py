import requests
import time
import typing
from threading import Event, Thread
from queue import Empty, Queue

from collections import defaultdict
from tritonclient import grpc as triton


class ServerInferenceMetric:
    def __init__(self):
        self.ns = 0
        self.count = 0

    def update(self, count, ns):
        update = count - self.count
        if update == 0:
            return
        average = (ns - self.ns) / update

        self.count = count
        self.ns = ns
        return average, update


class UnstableQueueException(Exception):
    pass


class ThreadedStatWriter(Thread):
    def __init__(
        self,
        output_file: str,
        columns: typing.List[str]
    ) -> None:
        self.output_file = output_file
        self.columns = columns
        self.f = None

        self._stop_event = Event()
        self._error_q = Queue()
        super().__init__()

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def error(self) -> Exception:
        try:
            return self._error_q.get_nowait()
        except Empty:
            return None

    def write_row(self, values):
        if len(values) != len(self.columns):
            raise ValueError(
                "Can't write values {} with length {}, "
                "doesn't match number of columns {}".format(
                    ", ".join(values), len(values), len(self.columns)
                )
            )
        self.f.write("\n" + ",".join(map(str, values)))

    def run(self):
        with open(self.output_file, "w") as self.f:
            self.f.write(",".join(self.columns))
            while not self.stopped:
                try:
                    values = self._get_values()
                    if values is not None:
                        if not isinstance(values[0], list):
                            values = [values]
                        for v in values:
                            self.write_row(v)
                except UnstableQueueException:
                    self.stop()
                except Exception as e:
                    self.stop()
                    self._error_q.put(e)

    def _get_values(self):
        raise NotImplementedError


class ServerStatsMonitor(ThreadedStatWriter):
    def __init__(
        self,
        output_file: str,
        url: str,
        model_name: str,
        qps_limit: float = 1000.,
        monitor: typing.Optional[str] = None,
        limit: typing.Optional[float] = None
    ):
        client = triton.InferenceServerClient(url)
        model_config = client.get_model_config(model_name).config
        if len(model_config.ensemble_scheduling.step) > 0:
            self.models = [
                i.model_name for i in model_config.ensemble_scheduling.step
            ]
        else:
            self.models = [model_config.name]
        self.url = "http://" + url.replace("8001", "8002/metrics")

        if monitor is not None and limit is None:
            raise ValueError(f"Must set limit for monitoring metric {monitor}")
        elif monitor is None:
            monitor = []
        elif isinstance(monitor, str):
            monitor = [monitor]

        self.monitor = monitor
        self.limit = limit

        self.stats = defaultdict(lambda: defaultdict(ServerInferenceMetric))
        self.qps_limit = qps_limit
        self._last_request_time = time.time()

        processes = [
            "success", "queue", "compute_input", "compute_infer", "compute_output"
        ]
        super().__init__(
            output_file, columns=["model"] + processes + ["count"]
        )

    def _get_values(self):
        values = []
        content = requests.get(self.url).content.decode("utf-8").split("\n")
        for model in self.models:
            model_stats = [i for i in content if f'model="{model}",' in i]
            if len(model_stats) == 0:
                raise ValueError("couldn't find model in content")

            counts = [
                float(i.split(" ")[1]) for i in model_stats if i.startswith(
                    "nv_inference_request_success"
                )
            ]
            count = sum(counts)

            model_values = [model]
            for process in self.columns[1:-1]:
                field = process if process != "success" else "request"
                process_times = [
                    float(i.split(" ")[1]) for i in model_stats if i.startswith(
                        f"nv_inference_{field}_duration_us"
                    )
                ]
                ns = sum(process_times)
                data = self.stats[model][process].update(count, ns)
                if data is None:
                    return

                t, diff = data
                model_values.append(t)

                if model in self.monitor and process == "queue":
                    if t > self.limit:
                        self._error_q.put(t)
                        raise UnstableQueueException

            model_values.append(diff)
            values.append(model_values)

        self._last_request_time = time.time()
        return values


class ClientStatsMonitor(ThreadedStatWriter):
    def __init__(self, output_file: str, q: Queue):
        self.q = q
        self.n = 0
        self._latency = 0

        self._throughput_q = Queue()
        self._request_rate_q = Queue()
        self._latency_q = Queue()

        super().__init__(
            output_file,
            columns=["message_start", "data_syncd", "request_send", "request_return"]
        )

    def _get_values(self):
        try:
            measurements = self.q.get_nowait()
        except Empty:
            return

        if measurements[0] == "start_time":
            self.start_time = measurements[1]
            return

        measurements = [i - self.start_time for i in measurements]
        self.n += 1
        self._throughput_q.put(self.n / measurements[-1])
        self._request_rate_q.put(self.n / measurements[1])

        latency = measurements[-1] - measurements[0]
        self._latency += (latency - self._latency) / self.n
        self._latency_q.put(self._latency)

        return measurements

    @property
    def latency(self):
        value = 0
        for i in range(100):
            try:
                value = self._latency_q.get_nowait()
            except Empty:
                break
        return int(value * 10**6)

    @property
    def throughput(self):
        value = 0
        for i in range(100):
            try:
                value = self._throughput_q.get_nowait()
            except Empty:
                break
        return value

    @property
    def request_rate(self):
        value = 0
        for i in range(100):
            try:
                value = self._request_rate_q.get_nowait()
            except Empty:
                break
        return value

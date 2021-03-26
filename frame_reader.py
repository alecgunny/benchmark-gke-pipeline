import time
import typing

from io import BytesIO
from threading import Event, Thread
from queue import Empty, Queue

import numpy as np
from google.cloud import storage
from gwpy.timeseries import TimeSeriesDict

from stillwater.data_generator import DataGenerator
from stillwater.utils import ExceptionWrapper, Package


class _RaisedFromParent(Exception):
    """
    Dummmy exception for breaking out of outer while loop
    below when in the while loop checking for q fullness
    """
    pass


class FrameBlobReader(Thread):
    def __init__(
        self,
        bucket_name: str,
        sample_rate: float,
        channels: typing.List[str],
        prefix=None
    ):
        client = storage.Client()
        try:
            bucket = client.get_bucket(bucket_name)
        except Exception as e:
            try:
                if e.code == 404:
                    raise ValueError(f"Couldn't find bucket {bucket_name}")
                raise
            except AttributeError:
                raise e
        self._blob_iterator = bucket.list_blobs(prefix=prefix)

        self.sample_rate = sample_rate
        self.channels = channels

        self._q = Queue(maxsize=1)
        self._stop_event = Event()
        super().__init__()

    def stop(self):
        self._stop_event.set()

    def get_next_frame(self):
        frame = self._q.get_nowait()
        if isinstance(frame, ExceptionWrapper):
            frame.reraise()
        return frame

    def run(self):
        try:
            for blob in self._blob_iterator:
                if self._stop_event.is_set():
                    break

                if not blob.name.endswith(".gwf"):
                    continue

                blob_bytes = BytesIO(blob.download_as_bytes())
                timeseries = TimeSeriesDict.read(
                    blob_bytes, channels=self.channels
                )
                timeseries.resample(self.sample_rate)

                frame = np.stack(
                    [timeseries[channel].value for channel in self.channels]
                )

                # don't sit and wait on the q.put in case
                # something happens in the parent thread
                # and we need to close out
                while self._q.full():
                    if self._stop_event.is_set():
                        raise _RaisedFromParent
                self._q.put(frame)
        except _RaisedFromParent:
            pass
        except Exception as e:
            self._q.put(ExceptionWrapper(e))


class GCPFrameDataGenerator(DataGenerator):
    def __init__(
        self,
        bucket_name: str,
        sample_rate: float,
        channels: typing.List[str],
        kernel_stride: float,
        generation_rate: typing.Optional[float] = None,
        prefix: typing.Optional[str] = None
    ):
        self._frame_reader = FrameBlobReader(
            bucket_name, sample_rate, channels, prefix
        )
        if generation_rate is not None:
            self._sleep_time = 1. / generation_rate - 2e-4
        else:
            self._sleep_time = None
        self._last_time = time.time()

        self._frame = None
        self._idx = 0
        self._step = int(kernel_stride * sample_rate)

    def __iter__(self):
        self._frame_reader.start()
        return self

    def __next__(self):
        self._idx += 1
        if (
            self._frame is None or
            (self._idx + 1) * self._step > self._frame.shape[1]
        ):
            while True:
                try:
                    frame = self._frame_reader.get_next_frame()
                    break
                except Empty:
                    if not self._frame_reader.is_alive():
                        raise StopIteration
                    continue

            if self._idx * self._step < self._frame.shape[1]:
                leftover = self._frame[:, self._idx * self._step:]
                frame = np.concatenate([leftover, frame], axis=1)

            self._frame = frame
            self._idx = 0

        if self._sleep_time is not None:
            while (time.time() - self._last_time) < self._sleep_time:
                time.sleep(1e-6)

        x = self._frame[self._idx * self._step: (self._idx + 1) * self._step]
        package = Package(x=x, t0=time.time())
        self._last_time = package.t0
        return package

    def stop(self):
        self._frame_reader.stop()
        self._frame_reader.join()

import time
import typing

from io import BytesIO
from multiprocessing import Event, Process, Queue
from queue import Empty

import numpy as np
from google.cloud import storage
from google.oauth2 import service_account
from gwpy.timeseries import TimeSeriesDict

from stillwater.data_generator import DataGenerator
from stillwater.utils import ExceptionWrapper, Package


class _RaisedFromParent(Exception):
    """
    Dummmy exception for breaking out of outer while loop
    below when in the while loop checking for q fullness
    """
    pass


def read_frames(
    service_account_key_file,
    q,
    stop_event,
    bucket_name,
    sample_rate,
    channels,
    prefix=None
):
    try:
        credentials = service_account.Credentials.from_service_account_file(
            service_account_key_file
        )
        client = storage.Client(credentials=credentials)
        try:
            bucket = client.get_bucket(bucket_name)
        except Exception as e:
            try:
                if e.code == 404:
                    raise ValueError(f"Couldn't find bucket {bucket_name}")
                raise
            except AttributeError:
                raise e

        for blob in bucket.list_blobs(prefix=prefix):
            print(blob.name)
            if stop_event.is_set():
                break

            if not blob.name.endswith(".gwf"):
                continue

            blob_bytes = GWFBytes(blob.download_as_bytes())

            timeseries = TimeSeriesDict.read(
                blob_bytes, channels=channels, format="gwf"
            )
            timeseries.resample(sample_rate)

            frame = np.stack(
                [timeseries[channel].value for channel in channels]
            )

            # don't sit and wait on the q.put in case
            # something happens in the parent thread
            # and we need to close out
            while q.full():
                if stop_event.is_set():
                    raise _RaisedFromParent
            q.put(frame)

    except _RaisedFromParent:
        pass
    except Exception as e:
        q.put(ExceptionWrapper(e))


class GCPFrameDataGenerator(DataGenerator):
    def __init__(
        self,
        credentials,
        bucket_name: str,
        sample_rate: float,
        channels: typing.List[str],
        kernel_stride: float,
        generation_rate: typing.Optional[float] = None,
        prefix: typing.Optional[str] = None
    ):
        self.bucket_name = bucket_name
        self.sample_rate = sample_rate
        self.channels = channels
        self.prefix = prefix
        self.service_account_key_file = credentials

        if generation_rate is not None:
            self._sleep_time = 1. / generation_rate - 2e-4
        else:
            self._sleep_time = None
        self._last_time = time.time()

        self._frame = None
        self._idx = 0
        self._step = int(kernel_stride * sample_rate)

    def __iter__(self):
        self._q = Queue(maxsize=1)
        self._stop_event = Event()
        self._frame_reader = Process(
            target=read_frames,
            args=(
                self.service_account_key_file,
                self._q,
                self._stop_event,
                self.bucket_name,
                self.sample_rate,
                self.channels,
                self.prefix
            )
        )
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
                    frame = self._q.get_nowait()
                    break
                except Empty:
                    if not self._frame_reader.is_alive():
                        raise StopIteration
                    continue
            if isinstance(frame, ExceptionWrapper):
                frame.reraise()

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
        self._stop_event.set()
        self._frame_reader.join(0.5)
        try:
            self._frame_reader.close()
        except ValueError:
            self._frame_reader.terminate()


if __name__ == "__main__":
    channels = """
H1:GDS-CALIB_F_CC_NOGATE
H1:GDS-CALIB_STATE_VECTOR
H1:GDS-CALIB_SRC_Q_INVERSE
H1:GDS-CALIB_KAPPA_TST_REAL_NOGATE
H1:GDS-CALIB_F_CC
H1:GDS-CALIB_KAPPA_TST_IMAGINARY
H1:IDQ-PGLITCH_OVL_32_2048
H1:GDS-CALIB_KAPPA_PU_REAL_NOGATE
H1:IDQ-LOGLIKE_OVL_32_2048
H1:GDS-CALIB_KAPPA_TST_REAL
H1:ODC-MASTER_CHANNEL_OUT_DQ
H1:GDS-CALIB_F_S_NOGATE
H1:GDS-CALIB_KAPPA_C
H1:IDQ-FAP_OVL_32_2048
H1:GDS-CALIB_SRC_Q_INVERSE_NOGATE
H1:GDS-CALIB_KAPPA_PU_REAL
H1:IDQ-EFF_OVL_32_2048
H1:GDS-CALIB_KAPPA_TST_IMAGINARY_NOGATE
H1:GDS-CALIB_KAPPA_PU_IMAGINARY
H1:GDS-CALIB_KAPPA_PU_IMAGINARY_NOGATE
H1:GDS-GATED_DQVECTOR
""".split("\n")

    class GWFBytes(BytesIO):
        name="banana.gwf"

    import glob
    dg = GCPFrameDataGenerator(
        "/home/alec.gunny/.ssh/gcp-service-account-key.json",
        # glob.glob(r"C:\Users\amacg\Downloads\gunny*.json")[0],
        bucket_name="ligo-o2",
        sample_rate=1000,
        channels=[i for i in channels if i],
        kernel_stride=0.1,
        generation_rate=1000,
        prefix="archive/frames/O2/hoft_C02/H1/H-H1_HOFT_C02-11854"
    )

    start_time = time.time()

    try:
        dg = iter(dg)
        n = 0
        while True:
            x = next(dg)
            n += 1

            throughput = n / (time.time() - start_time)
            msg = f"Output rate: {throughput:0.1f}"
            print(msg, end="\r", flush=True)

            if n == 10000:
                break
    finally:
        dg.stop()

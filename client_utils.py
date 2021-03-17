import logging
import queue
import time
import typing

import attr

from stillwater import sync_recv
from stillwater.data_generator import LowLatencyFrameGenerator

if typing.TYPE_CHECKING:
    from multiprocessing import Process
    from multiprocessing.connection import Connection


# set up logger
log = logging.getLogger()
console = logging.StreamHandler()
console.setFormatter(
    logging.Formatter("%(asctime)s\t%(message)s")
)
log.addHandler(console)
log.setLevel(logging.INFO)


@attr.s(auto_attribs=True)
class Pipeline:
    processes: typing.List["Process"]
    out_pipes: typing.Dict[str, "Connection"]

    def cleanup(self):
        for process in self.processes:
            try:
                is_alive = process.is_alive()
            except ValueError:
                continue
            if is_alive:
                process.stop()
                process.join(0.5)

            try:
                process.close()
            except ValueError:
                process.terminate()
                time.sleep(0.1)
                process.close()
                print(f"Process {process.name} couldn't join")

    def get(self, timeout=None):
        try:
            package = sync_recv(self.out_pipes, timeout=timeout)
        except Exception:
            self.cleanup()
            raise
        return package

    def reset(self):
        log.info("Pausing processes")
        for process in self.processes:
            process.pause()

        log.info("Clearing out pipes")
        for pipe in self.out_pipes.values():
            while pipe.poll():
                _ = pipe.recv()

        log.info(
            "Resetting, clearing metric qs and unpausing processes"
        )
        t0 = None
        for process in self.processes:
            log.info(f"Resetting process {process.name}")
            if isinstance(process, LowLatencyFrameGenerator):
                process._in_q.put(("reset", {"t0": t0}))
                process._in_q.join()

                while True:
                    try:
                        t0 = process._in_q.get_nowait()
                        process._in_q.task_done()
                        log.info(f"Process {process.name} t0 now {t0}")
                        break
                    except queue.Empty:
                        continue
            else:
                process._in_q.put("reset")
                process._in_q.join()

            while True:
                try:
                    _ = process._metric_q.get_nowait()
                except queue.Empty:
                    break

            log.info(f"Unpausing process {process.name}")
            process.resume()

    def __enter__(self):
        for process in self.processes:
            if not process.is_alive() and process.exitcode is None:
                process.start()
            # or else what?
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.cleanup()

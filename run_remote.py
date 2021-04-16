import argparse
import os
import subprocess
import time
import typing

from google.oauth2 import service_account


def run_cmd(cmd, verbose=False):
    if verbose:
        print(cmd)
    result = subprocess.run(
        cmd, capture_output=True, check=True, shell=True
    ).stdout.decode("utf-8")
    if verbose:
        print(result)
    return result


def main(
    service_account_key_file: str,
    vm_name: str,
    ip_address: str,
    ssh_key_file: str,
    start: int,
    stop: int,
    step: int,
    output_dir: typing.Optional[str] = None,
    num_retries: typing.Optional[int] = 0
) -> None:
    output_dir = output_dir or "."
    credentials = service_account.Credentials.from_service_account_file(
        service_account_key_file
    )

    service_account_email = credentials._service_account_email
    project = credentials._project_id
    base_cmd = _get_base_cmd(
        vm_name,
        "alecgunny/gw-client:latest",
        service_account_email,
        project,
        32
    )

    generation_rate = start
    num_clients = 1
    current_retries = 0
    while True:
        # build and run the command that executes the job on GCP
        client_cmd = _get_client_cmd(
            ip_address,
            generation_rate,
            num_iterations=50000,
            num_clients=num_clients,
            latency_threshold=1.0,
            queue_threshold=100000,
            num_retries=0
        )
        client_cmd = [f"--container-arg={i}" for i in client_cmd.split()]
        cmd = base_cmd + " ".join(client_cmd)
        run_cmd(cmd, True)

        # sleep for a bit to give it time to spin up
        time.sleep(10)

        try:
            _wait_for_container_completion(vm_name, project, ssh_key_file)
            _copy_results(
                vm_name,
                project,
                ssh_key_file,
                generation_rate,
                num_clients,
                output_dir
            )
        except RuntimeError as e:
            # if the container never started, try to pull
            # pull a log file and see what happened
            try:
                cmd = _get_scp_cmd(
                    "output.log",
                    vm_name,
                    project,
                    ssh_key_file,
                    generation_rate,
                    num_clients,
                    output_dir
                )
                run_cmd(cmd, True)
            except Exception:
                # there was no log file, guess we're out of luck
                pass
            else:
                # read the file and print it to stdout, then get rid of it
                prefix = "generation-rate={}_clients={}".format(
                    generation_rate, num_clients
                )
                fname = os.path.join(output_dir, prefix + "_output.log")
                with open(fname, "r") as f:
                    log = f.read()
                    print(log)
                os.remove(fname)

            finally:
                if "[StatusCode.UNAVAILABLE] Too many pings" in log:
                    # having inconsistent issues with connecting
                    # to the server, so try this a couple times
                    # before calling it quits
                    current_retries += 1
                    if current_retries < num_retries:
                        continue
                # otherwise raise the original error that
                # got us here in the first place
                raise e
        finally:
            # delete the vm whether we succeeded or not
            cmd = _get_delete_cmd(vm_name, project)
            run_cmd(cmd, True)

        # load in the log from the run and make sure
        # we didn't run into any issues
        prefix = "generation-rate={}_clients={}".format(
            generation_rate, num_clients
        )
        fname = os.path.join(output_dir, prefix + "_output.log")
        with open(fname, "r") as f:
            log = f.read()

        bad_messages = [
            "Queue times stable, retrying",
            "[StatusCode.UNAVAILABLE] Too many pings",
            "[StatusCode.UNAVAILABLE] Broken pipe"
        ]
        try:
            for msg in bad_messages:
                if msg in log:
                    raise RuntimeError(msg)
            else:
                current_retries = 0
        except RuntimeError as e:
            current_retries += 1
            if current_retries == num_retries:
                raise RuntimeError("Too many retries")
            print("Retrying due to message: " + str(e))
            continue

        if stop is not None:
            generation_rate += step
            if generation_rate >= stop:
                break
        else:
            prefix = "generation-rate={}_clients={}".format(
                generation_rate, num_clients
            )
            fname = os.path.join(output_dir, prefix + "_output.log")
            with open(fname, "r") as f:
                log = f.read()
            if "MonitoredMetricViolationException" in log:
                if "snapshotter_queue" in log:
                    num_clients += 1
                else:
                    return generation_rate, num_clients
            else:
                generation_rate += step


def _wait_for_container_completion(vm_name, project, ssh_key_file):
    container_started = False
    start_up_sleep = 60
    total_wait_time = 300
    start_time = time.time()
    while True:
        try:
            cmd = _get_ssh_cmd(vm_name, project, ssh_key_file)
            result = run_cmd(cmd, verbose=False)
        except subprocess.CalledProcessError as e:
            if time.time() - start_time > start_up_sleep:
                raise RuntimeError(e.stderr.decode("utf-8"))
            continue

        if "alecgunny/gw-client:latest" not in result:
            if not container_started:
                if time.time() - start_time > start_up_sleep:
                    raise RuntimeError(
                        "Container hasn't started after {} seconds, "
                        "`docker ps` returned:\n{}".format(
                            start_up_sleep + 10, result
                        )
                    )
            else:
                break
        else:
            if time.time() - start_time > total_wait_time:
                raise RuntimeError("Job taking too long, stopping")
            container_started = True


def _copy_results(
    vm_name,
    project,
    ssh_key_file,
    generation_rate,
    num_clients,
    output_dir
):
    for fname in ["output.log", "server-stats.csv", "client-stats.csv"]:
        cmd = _get_scp_cmd(
            fname,
            vm_name,
            project,
            ssh_key_file,
            generation_rate,
            num_clients,
            output_dir
        )
        try:
            run_cmd(cmd, True)
        except subprocess.CalledProcessError as e:
            print(e.stderr)
            continue


def _get_delete_cmd(name, project):
    cmd = f"gcloud compute instances delete {name} --project {project} --quiet"
    return cmd


def _get_scp_cmd(
    remote_fname,
    name,
    project,
    ssh_key_file,
    generation_rate,
    num_clients,
    output_dir
):
    # TODO: add username
    prefix = f"generation-rate={generation_rate}_clients={num_clients}"
    cmd = f"gcloud compute scp --project {project} "
    cmd += f"--ssh-key-file {ssh_key_file} "
    cmd += f"alec.gunny@{name}:/home/{remote_fname} "
    cmd += os.path.join(output_dir, f"{prefix}_{remote_fname}")
    return cmd


def _get_ssh_cmd(name, project, ssh_key_file):
    # TODO: add username
    cmd = f"gcloud compute ssh alec.gunny@{name} --project {project} "
    cmd += f"--ssh-key-file {ssh_key_file}"
    cmd = cmd + ' --command="docker ps"'
    return cmd


def _get_base_cmd(
    name,
    container_image,
    service_account_email,
    project,
    num_vcpus
):
    cmd = f"gcloud compute instances create-with-container {name} "
    cmd += f"--container-image {container_image} "
    cmd += f"--machine-type n1-highcpu-{num_vcpus} "
    cmd += f"--service-account {service_account_email} "
    cmd += f"--project {project} "
    cmd += "--container-mount-host-path=host-path=/home,mount-path=/output "
    cmd += "--container-restart-policy=never "
    cmd += '--min-cpu-platform="Intel Skylake" '
    return cmd


def _get_client_cmd(
    ip_address,
    generation_rate,
    num_iterations,
    num_clients,
    latency_threshold,
    queue_threshold,
    num_retries
):
    cmd = f"--url {ip_address}:8001 "
    cmd += "--model-name gwe2e --model-version 1 --sequence-id 1001 "
    cmd += f"--generation-rate {generation_rate} "
    cmd += f"--num-iterations {num_iterations} "
    cmd += f"--num-clients {num_clients} "
    cmd += "--file-prefix /output/ --log-file /output/output.log --warm-up 10 "
    cmd += f"--latency-threshold {latency_threshold} "
    cmd += f"--queue-threshold-us {queue_threshold} "
    cmd += f"--num-retries {num_retries}"
    return cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--service-account-key-file",
        type=str,
        required=True,
        help="Path to GCP service account key json"
    )
    parser.add_argument(
        "--vm-name",
        type=str,
        default="gw-client-1",
        help="Name to give client VM instance"
    )
    parser.add_argument(
        "--ip-address",
        type=str,
        required=True,
        help="IP address of Triton server"
    )
    parser.add_argument(
        "--ssh-key-file",
        type=str,
        required=True,
        help="Path to ssh key for connecting to instance"
    )
    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Initial generation rate"
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help=(
            "Final generation rate. If `None`, run until "
            "a metric violates its threshold limit."
        )
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Interval at which to increase generation rate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="where to write results to"
    )
    parser.add_argument(
        "--num-retries",
        type=int,
        default=0,
        help="Number of retry attempts"
    )

    flags = parser.parse_args()
    main(**vars(flags))

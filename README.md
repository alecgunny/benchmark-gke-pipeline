# End-to-End Benchmarking Pipeline for Gravitational Wave DL Inference

## Requirements
These instructions will assume that you have a conda installation on your devices. If you don't, follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) (and in particular, consider using the Miniconda installer for environments where you don't want or need all the packages that Anaconda ships with).

### Locally
On your local device, you will manage GKE cluster resources and perform the export of any models as necessary. As such, you will need:
1. The export conda environment
```
conda env create -f conda/environment.export.yaml
```
2. The gcloud command line utility, installation instructions [here](https://cloud.google.com/sdk/docs/install). Be sure that you have the appropriate permissions to create and delete clusters and nodepools.
3. The gsutil command line utility, installation instructions [here](https://cloud.google.com/storage/docs/gsutil_install). Be sure that you have the appropriate permissions to create and delete buckets and blobs.
4. The Kubernetes client, installation instructions for Linux [here](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/#install-kubectl-binary-with-curl-on-linux) (consider following the local install instructions if you don't have root access on your "local" node), and links for installation on other platforms [here](https://kubernetes.io/docs/tasks/tools/)

### Remotely
Your remote environment will need to have the client conda environment built, either by building locally and exporting as a tar archive or by building there:
```
conda env create -f conda/environment.client.yaml
```

## Running
Before you do anything, feel free to run the various bash scripts with the `-h` flag to get a sense for what you can control and what each script does. In the example below, all the CAPITALIZED variables are arbitrarily chosen and can be edited to be whatever you want, and anything in `<brackets>` indicates something that you'll need to fill in which is specific to your account settings.

### Spin up a cluster
Start by spinning up a cluster on which our tests will run. We'll do this before model export because if we want to leverage TensorRT (TRT) at inference time, we'll host a TRT conversion app on this cluster which will run a pre-built Docker container on our GPU platform of choice (T4 in this case). This way, you don't need local GPU access to run things, which is a plus if you want to use enterprise-grade GPUs.

```
CLUSTER_NAME=gw-benchmarking
ZONE=us-west1-b
PROJECT=<your GCP project name>
./manage-cluster.sh create -c $CLUSTER_NAME -z $ZONE -p $PROJECT
```

### Export
Before we export, be sure to activate the export conda environment
```
conda activate gwe2e-export
```

Now exporting the relevant models locally then ship them to a GCS bucket. If you only ever plan on leveraging a single kernel stride, you'll only need to do this once up front. Note that if the bucket you point to doesn't exist in your project, the script will attempt to create one by that name. GCS bucket names must be _globally_ unique: that is, buckets of any given name can only exist once on _any_ project _anywhere_, so be sure to come up with a unique name or the script will error out if it attempts to create a bucket that already exists.
```
KERNEL_STRIDE=0.002
STREAMS_PER_GPU=2
INSTANCES_PER_GPUS=2
LOCAL_REPO_NAME=./repo
BUCKET_NAME=gw-benchmarking_model-repo

# the -t flag indicates to use TensorRT
# the -d flag indicates to delete the local model repo once we've
#    copied the models to GCS
# the -f flag indicates to use FP16 precision for the TensorRT models

./export.sh -c $CLUSTER_NAME -z $ZONE -p $PROJECT \
    -r $LOCAL_REPO_NAME -b $BUCKET_NAME \
    -k $KERNEL_STRIDE -i $INSTANCES_PER_GPU \
    -s $STREAMS_PER_GPU -t -f -d
```

### Start the server and deloy the clients locally
For **_local_** client deployment, activate the client conda environment
```
conda activate gwe2e-client
```

Then run `client.sh`, which will spin up server nodes as well as
local clients for benchmarking

```
NUM_GPUS=2
VCPUS_PER_GPU=16
NUM_NODES=1
DATA_GENERATRION_RATE=1200
ITERATIONS=50000
LATENCY_THRESHOLD_SECONDS=1
Q_THRESHOLD_MICROSECONDS=100000

./client.sh -c $CLUSTER_NAME -p $PROJECT -z $ZONE \
    -b $BUCKET_NAME -G $NUM_GPUS -g $NUM_GPUS -v $VCPUS_PER_GPU \
    -N $NUM_NODES -r $DATA_GENERATION_RATE -i $ITERATIONS \
    -q $Q_THRESHOLD_MICROSECONDS -l $LATENCY_THRESHOLD_SECONDS
```

This will profile an inference run and dump the measurements into csvs `node-<i>_<server|client>-stats.csv` and any logs or erros into `node-<i>_output.log`, where `i` indexes each client instance.


### Multi-client deployment
In this case, you'll need to start by spinning up multiple servers yourself from your local node using a loop like at the top of `client.sh`:
```
# note that you'll probably want to create multiple nodes
# if you want to use multiple clients, since one client
# is more than capable of saturating at least one node,
# if not more
NUM_NODES=<some value greater than 1>

# also note that only the first iteration actually creates
# a node pool - after the first run we detect that a node
# pool already exists and just ignore it and create the
# new deployment/service with a new name to get a new node
for i in $(seq $NUM_NODES); do
    ./start-server.sh -c $CLUSTER_NAME -p $PROJECT -z $ZONE \
        -b $BUCKET_NAME -G $NUM_GPUS -g $NUM_GPUS \ 
        -v $VCPUS_PER_GPU -N $NUM_NODES -n "tritonserver-${i}"
done
```

Then you'll need to build a loop that looks a lot like the one at the bottom of `client.sh`, where you start a client instance for each IP address:
```
for i in $(seq $NUM_NODES); do
    <whatever your command is that spins up a client node>

    # make sure this command runs asynchronously so
    # that all the clients run simultaneously

    name="tritonserver-${i}"
    ip=$(kubectl get service $name -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    <some remote execution command> python client.py \
            --url $ip:8001 \
            --model-name gwe2e \
            --model-version 1 \
            --generation-rate ${rate} \
            --num-iterations ${iterations} \
            --warm-up 10 \
            --file-prefix "node-${i}" \
            --queue-threshold-us ${queue} \
            --latency-threshold ${latency}
```
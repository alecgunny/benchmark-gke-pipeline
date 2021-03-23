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


### Start the Triton Server
Create a T4 node pool then deploy the Triton server container on to it, pointing it at our export bucket.
```
NUM_GPUS=2
NUM_VCPUS=32
NUM_NODES=1
./start-server.sh -c $CLUSTER_NAME -z $ZONE -p $PROJECT \
    -b $BUCKET_NAME -g $NUM_GPUS -v $NUM_VCPUS -N 1
```

### Deploy the client nodes and run experiment
This is the part that needs completing.

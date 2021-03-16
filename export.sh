#! /bin/bash -e

while getopts "c:p:z:r:b:k:i:s:tdf" opt; do
    case ${opt} in
        c )
            cluster=${OPTARG}
            ;;
        p )
            project=${OPTARG}
            ;;
        z )
            zone=${OPTARG}
            ;;
        r )
            repo=${OPTARG}
            ;;
        b )
            bucket=${OPTARG}
            ;;
        k )
            kernel_stride=${OPTARG}
            ;;
        i )
            instances=${OPTARG}
            ;;
        s )
            streams=${OPTARG}
            ;;
        t )
            trt=true
            ;;
        d )
            delete=true
            ;;
        f ) use_fp16="--use-fp16"
            ;;
        h )
            echo "Create or delete a GKE cluster"
            echo "Options:"
            echo "--------"
            echo "    -c: cluster name for deploying TensorRT conversion app"
            echo "    -p: project name for deploying TensorRT conversion app"
            echo "    -z: zone for deploying TensorRT conversion app"
            echo "    -r: local repository to save exported models to"
            echo "    -b: GCP bucket to which to host models after export"
            echo "    -k: kernel stride"
            echo "    -i: number of instances per model per gpu"
            echo "    -s: streams per node"
            echo "    -t: set this flag to convert deepclean models to TensorRT"
            echo "    -d: set this flag to delete local repository after export to GCP"
            echo "    -f: set this flag to use fp16 inference with TensorRT"
            exit 0
            ;;
        \? )
            echo "Unrecognized argument ${opt}"
            exit 1
    esac
done
shift $((OPTIND -1))

: ${project:?Must specify GCP project name}

if [[ ! -z $trt ]]; then
    : ${cluster:?Must specify cluster name for TensorRT conversion}
    : ${zone:?Must specify GCP zone for TensorRT conversion}

    # create the node pool for the TRT converter app
    ./manage-node-pool.sh create \
        -n trt-converter-pool \
        -c ${cluster} \
        -p ${project} \
        -z ${zone} \
        -g 1 \
        -N 1 \
        -v 4 \
        -l trtconverter=true

    # deploy the app and wait for it to be ready
    kubectl apply -f apps/trt-converter/deploy.yaml
    kubectl rollout status deployment/trt-converter

    # get the IP of the load balancer ingress to make requests
    ip=$(kubectl get service trt-converter -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

    # attach it to the platform so our export script knows
    # not to try to perform a local conversion, include
    # the fp16 flag (or its absence) here
    platform="trt:${ip} ${use_fp16}"
else
    platform="onnx"
fi

# make the repo if it doesn't exist. If it does,
# delete all the contents first so we can start fresh
if [[ ! -d $repo ]]; then mkdir -p ${repo}; fi
if [[ ! -z $(ls $repo) ]]; then rm -rf ${repo}/*; fi

# run the export script
python export.py \
    --count ${instances} \
    --platform ${platform} \
    --kernel-stride ${kernel_stride}

# create the specified bucket if it doesn't exist
exists=$(gsutil ls -p ${project} gs:// | grep gs://${bucket})
if [[ -z $exists ]]; then
    gsutil mb -p ${project} gs://${bucket}
fi

# copy all the repo contents to the bucket
gsutil cp -p ${project} ${repo}/* gs://${bucket}

# delete the local contents if we set the -d flag
if [[ ! -z $delete ]]; then
    rm -rf ${repo}
fi

# delete the converter node pool now that
# we're done with it. In production, you
# probably wouldn't want to do this
if [[ ! -z $trt ]]; then
    ./manage-node-pool.sh delete \
        -n trt-converter-pool -c ${cluster} -p ${project} -z ${zone}
fi

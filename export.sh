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
        f ) precision="fp16"
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
            echo "    -s: streams per gpu"
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

# check required args
: ${project:?Must specify GCP project name}
: ${repo:?Must specify local export repo}
: ${kernel_stride:?Must specify kernel stride}

if [[ ! -z $trt ]]; then
    # check args required for doing TRT export
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
        -l trtconverter=true || echo "Conversion node pool already created"

    # deploy the app and wait for it to be ready
    kubectl apply -f apps/trt-converter/deploy.yaml || echo "Conversion service already created"
    kubectl rollout status deployment/trt-converter

    # get the IP of the load balancer ingress to make requests
    ip=$(kubectl get service trt-converter -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

    # attach it to the platform so our export script knows
    # not to try to perform a local conversion
    platform="trt_${precision:-fp32}:http://${ip}:5000/onnx"
else
    platform="onnx"
fi

# make the repo if it doesn't exist. If it does,
# delete all the contents first so we can start fresh
[[ -d $repo ]] || mkdir -p ${repo}
[[ -z $(ls $repo) ]] || rm -rf ${repo}/*

# run the export script
python export.py \
    --repo-dir ${repo} \
    --count ${instances:-1} \
    --platform ${platform} \
    --kernel-stride ${kernel_stride} \
    --streams-per-gpu ${streams:-1}

# create the specified bucket if it doesn't exist
[[ ! -z $(gsutil ls -p ${project} gs:// | grep gs://${bucket}) ]] || \
    gsutil mb -p ${project} gs://${bucket} || ("GCS bucket named ${bucket} already exists" && exit 1)

# copy all the repo contents to the bucket
gsutil cp -r ${repo}/* gs://${bucket}

# delete the local contents if we set the -d flag
[[ -z $delete ]] || rm -rf ${repo}

# delete the converter node pool now that
# we're done with it. In production, you
# probably wouldn't want to do this
if [[ ! -z $trt ]]; then
    kubectl delete -f apps/trt-converter/deploy.yaml
    ./manage-node-pool.sh delete \
        -n trt-converter-pool -c ${cluster} -p ${project} -z ${zone}
fi


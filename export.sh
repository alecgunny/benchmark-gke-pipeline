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


if [[ ! -z $trt ]]; then
    if [[ -z $cluster ]]; then
        echo "Must specify cluster for TensorRT conversion"
        exit 1
    elif [[ -z $project ]]; then
        echo "Must specify project for TensorRT conversion"
        exit 1
    elif [[ -z $zone ]]; then
        echo "Must specify zone for TensorRT conversion"
        exit 1
    fi

    ./manage-node-pool.sh create \
        -n trt-converter-pool \
        -c ${cluster} \
        -p ${project} \
        -z ${zone} \
        -g 1 \
        -N 1 \
        -v 4 \
        -l trtconverter=true

    kubectl apply -f apps/trt-converter/deploy.yaml
    kubectl rollout status deployment/trt-converter

    ip=$(kubectl get service trt-converter -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    platform="trt:${ip}"
else
    platform="onnx"
fi

if [[ ! -d $repo ]]; then mkdir -p ${repo}; fi
if [[ ! -z $(ls $repo) ]]; then rm -rf ${repo}/*; fi

python export.py \
    --count ${instances} \
    --platform ${platform} \
    --kernel-stride ${kernel_stride} \
    ${use_fp16}

gsutil cp ${repo}/* gs://${bucket}
if [[ ! -z $delete ]]; then
    rm -rf ${repo}
fi

if [[ ! -z $trt ]]; then
    ./manage-node-pool.sh \
        -m delete -c ${cluster} -p ${project}
fi


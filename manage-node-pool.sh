#! /bin/bash -e

while getopts ":n:c:p:m:g:N:v" opt; do
    case ${opt} in
        n )
            nodepool=${OPTARG}
            ;;
        c )
            cluster=${OPTARG}
            ;;
        p )
            project=${OPTARG}
            ;;
        m )
            mode=${OPTARG}
            ;;
        g )
            gpus=${OPTARG}
            ;;
        N)
            nodes=${OPTARG}
            ;;
        v)
            vcpus=${OPTARG}
            ;;
        h )
            echo "Create or delete a GKE cluster"
            echo "Options:"
            echo "--------"
            echo "    -n: nodepool name"
            echo "    -c: cluster name"
            echo "    -p: project name"
            echo "    -m: mode, either create or delete"
            echo "    -g: number of T4s to attach to nodes on nodepool"
            echo "    -N: number of nodes to add to nodepool"
            echo "    -v: number of vCPUs to attach to nodes on nodepool"
            exit 0
            ;;
        \? )
            echo "Unrecognized argument ${opt}"
            exit 1
    esac
done
shift $((OPTIND -1))

if [[ -z $nodepool ]]; then
    echo "Must specify a name for nodepool"
    exit 1
fi

if [[ -z $cluster ]]; then
    echo "Must specify a cluster to deploy nodepool ${nodepool} on"
    exit 1
fi

if [[ -z $project ]]; then
    echo "Must specify a project for nodepool ${nodepool}"
    exit 1
fi

if [[ -z $mode ]]; then
    echo "No mode specified!"
    exit 1
elif [[ $mode == "create" ]]; then
    gcloud container nodepool create ${nodepool} \
        --cluster=${cluster} \
        --project=${project} \
        --machine-type=n1-standard-${vcpus} \
        --accelerator=type=nvidia-tesla-t4,count=${gpus}
elif [[ $mode == "delete" ]]; then
    gcloud container nodepool delete ${nodepool} \
        --cluster=${cluster} \
        --project=${project}
else
    echo "Unrecognized mode ${mode}"
    exit 1
fi

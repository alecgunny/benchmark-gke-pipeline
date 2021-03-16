#! /bin/bash -e

while getopts "n:c:p:z:m:g:N:v:h" opt; do
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
        z )
            zone=${OPTARG}
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
            echo "    -z: zone to create node pool in"
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
elif [[ -z $cluster ]]; then
    echo "Must specify a cluster to deploy nodepool ${nodepool} on"
    exit 1
elif [[ -z $project ]]; then
    echo "Must specify a project for nodepool ${nodepool}"
    exit 1
elif [[ -z $zone ]]; then
    echo "Must specify a zone for nodepool ${nodepool}"
fi

if [[ -z $mode ]]; then
    echo "No mode specified!"
    exit 1
elif [[ $mode == "create" ]]; then
    gcloud container node-pools create ${nodepool} \
        --cluster=${cluster} \
        --project=${project} \
        --zone=${zone} \
        --machine-type=n1-standard-${vcpus} \
        --accelerator=type=nvidia-tesla-t4,count=${gpus}
elif [[ $mode == "delete" ]]; then
    gcloud container node-pools delete ${nodepool} \
        --cluster=${cluster} \
        --project=${project} \
        --zone=${zone}
else
    echo "Unrecognized mode ${mode}"
    exit 1
fi


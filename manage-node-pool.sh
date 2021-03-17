#! /bin/bash -e

print_help() {
    echo "Create or delete a GKE cluster"
    echo "Usage: ./manage-cluster.sh CMD [options]"
    echo "CMD: either create or delete"
    echo "Options:"
    echo "--------"
    echo "    -n: nodepool name"
    echo "    -c: cluster name"
    echo "    -p: project name"
    echo "    -z: zone to create node pool in"
    echo "    -g: number of T4s to attach to nodes on nodepool. Only necessary in create mode"
    echo "    -N: number of nodes to add to nodepool. Only necessary in create mode"
    echo "    -v: number of vCPUs to attach to nodes on nodepool. Only necessary in create mode"
    echo "    -l: optional labels to provide to node pool creation"
    exit 0
}

cmd=$1
error=""
if [[ $cmd == -* ]] || [[ -z $cmd ]]; then
    [[ $cmd != "-h" ]] || print_help
    error="Must provide command create or delete!"
elif [[ $cmd != "create" ]] && [[ $cmd != "delete" ]]; then
    error="Provided unrecognized command $cmd"
fi
shift

while getopts "n:c:p:z:g:N:v:l:h" opt; do
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
        g )
            gpus=${OPTARG}
            ;;
        N )
            nodes=${OPTARG}
            ;;
        v )
            vcpus=${OPTARG}
            ;;
        l )
            labels="--node-labels=${OPTARG}"
            ;;
        h )
            print_help
            ;;
        \? )
            echo "Unrecognized argument ${opt}"
            exit 1
    esac
done
shift $((OPTIND -1))

if [[ ! -z $error ]]; then
    echo $error
    exit 1
fi

: ${nodepool:?Must specify nodepool name}
: ${cluster:?Must specify cluster name}
: ${zone:?Must specify zone}
: ${project:?Must specify project}

if [[ $cmd == "create" ]]; then
    : ${gpus:?Must specify number of gpus to attach to nodes}
    : ${vcpus:?Must specify number of vcpus to attach to nodes}
    : ${nodes:?Must specify number of nodes to create in node pool}

    gcloud container node-pools create ${nodepool} \
        --verbosity=error \
        --cluster=${cluster} \
        --project=${project} \
        --zone=${zone} \
        --num-nodes=${nodes} \
        --machine-type=n1-standard-${vcpus} \
        --accelerator=type=nvidia-tesla-t4,count=${gpus} \
        ${labels}
elif [[ $cmd == "delete" ]]; then
    gcloud container node-pools delete ${nodepool} \
        --quiet \
        --cluster=${cluster} \
        --project=${project} \
        --zone=${zone}
fi

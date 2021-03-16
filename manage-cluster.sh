#! /bin/bash -e

cmd=$1
if [[ $cmd == -* ]] || [[ -z $cmd ]]; then
    echo "Must provide command create or delete!"
    exit 1
elif [[ $cmd != "create" ]] && [[ $cmd != "delete" ]]; then
    echo "Provided unrecognized command $cmd"
    exit 1
fi
shift

while getopts "c:z:p:h" opt; do
    case ${opt} in
        c )
            cluster=${OPTARG}
            ;;
        z )
            zone=${OPTARG}
            ;;
        p )
            project=${OPTARG}
            ;;
        h )
            echo "Create or delete a GKE cluster"
            echo "Usage: ./manage-cluster.sh CMD [options]"
            echo "CMD: either create or delete"
            echo "Options:"
            echo "--------"
            echo "    -c: cluster name"
            echo "    -z: zone"
            echo "    -p: GCP project name"
            exit 0
            ;;
        \? )
            echo "Unrecognized argument ${opt}"
            exit 1
            ;;
    esac
done
shift $((OPTIND -1))

: ${cluster:?Must specify cluster name}
: ${zone:?Must specify zone}
: ${project:?Must specify project}

if [[ $cmd == "create" ]]; then
    gcloud container clusters create ${cluster} --num-nodes=2 --project=${project} --zone=${zone}

    # add cluster credentials to k8s config
    gcloud container clusters get-credentials ${cluster} --project=${project} --zone=${zone}
elif [[ $cmd == "delete" ]]; then
    gcloud container clusters delete ${cluster} --zone=${zone} --project=${project}

    # also remove cluster credentials from kube config
    kubename=gke_${project}_${zone}_${cluster}
    kubectl config unset users.$kubename
    kubectl config unset contexts.$kubename
    kubectl config unset clusters.$kubename
fi

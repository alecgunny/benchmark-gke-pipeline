#! /bin/bash -e

while getopts ":c:m:z:p" opt; do
    case ${opt} in
        c )
            cluster=${OPTARG}
            ;;
        m )
            mode=${OPTARG}
            ;;
        z )
            zone=${OPTARG}
            ;;
        p )
            project=${OPTARG}
            ;;
        h )
            echo "Create or delete a GKE cluster"
            echo "Options:"
            echo "--------"
            echo "    -c: cluster name"
            echo "    -m: mode, create or delete"
            echo "    -z: zone"
            echo "    -p: GCP project name"
            exit 0
            ;;
        \? )
            echo "Unrecognized argument ${opt}"
            exit 1
    esac
done
shift $((OPTIND -1))

if [[ -z $zone ]]; then
    echo "No zone specified!"
    exit 1

if [[ -z $project ]]; then
    echo "No project specified!"
fi

if [[ -z $mode ]]; then
    echo "No mode specified!"
    exit 1
elif [[ $mode == "create" ]]; then
    gcloud container cluster create ${cluster} --num-nodes=2 --project=${project} --zone=${zone}
    gcloud container clusters get-credentials ${cluster} --project=${project} --zone=${zone}
elif [[ $mode == "delete" ]]; then
    gcloud container cluster delete ${cluster} --zone=${zone} --project=${project}

    kubename=gke_${project}_${zone}_${cluster}
    kubectl config unset users.$kubename
    kubectl config unset contexts.$kubename
    kubectl config unset clusters.$kubename
else
    echo "Unrecognized mode ${mode}"
    exit 1
fi

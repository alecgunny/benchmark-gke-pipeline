#! /bin/bash -e

while getopts "c:p:z:b:g:v:N:h" opt; do
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
        b )
            bucket=${OPTARG}
            ;;
        g )
            gpus=${OPTARG}
            ;;
        v )
            vcpus=${OPTARG}
            ;;
        N )
            nodes=${OPTARG}
            ;;
        h )
            echo "Create or delete a GKE cluster"
            echo "Options:"
            echo "--------"
            echo "    -c: cluster name for deploying Triton server"
            echo "    -p: project name for deploying Triton server"
            echo "    -z: zone for deploying Triton server"
            echo "    -b: GCP bucket for hosting repo"
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

./manage-node-pool.sh create \
        -n tritonserver-pool \
        -c ${cluster} \
        -p ${project} \
        -z ${zone} \
        -g ${gpus} \
        -N ${nodes} \
        -v ${vcpus} || echo "Triton node pool already created" 2>/dev/null

python format_yaml.py apps/triton-server/deploy.yaml \
    --gpus $gpus --vcpus $(($vcpus - 1)) --repo $bucket | kubectl apply -f -
kubectl rollout status deployment/tritonserver

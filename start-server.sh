#! /bin/bash -e

while getopts "c:p:z:b:g:G:v:N:n:h" opt; do
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
            gpus_per_deployment=${OPTARG}
            ;;
        G )
            gpus_per_node=${OPTARG}
            ;;
        v )
            vcpus_per_gpu=${OPTARG}
            ;;
        N )
            nodes=${OPTARG}
            ;;
        n )
            name=${OPTARG}
            ;;
        h )
            echo "Create or delete a GKE cluster"
            echo "Options:"
            echo "--------"
            echo "    -c: cluster name for deploying Triton server"
            echo "    -p: project name for deploying Triton server"
            echo "    -z: zone for deploying Triton server"
            echo "    -b: GCP bucket for hosting repo"
            echo "    -G: number of T4s to attach to nodes on nodepool"
            echo "    -g: number of T4s to attach to nodes on deployment"
            echo "    -N: number of nodes to add to nodepool"
            echo "    -n: name for the Triton deployment"
            echo "    -v: number of vCPUs to attach to nodes and deployment per GPU"
            exit 0
            ;;
        \? )
            echo "Unrecognized argument ${opt}"
            exit 1
    esac
done
shift $((OPTIND -1))

[[ $gpus_per_node -ge $gpus_per_deployment ]] || echo "Too many gpus per deployment!"

vcpus_per_node=$(($vcpus_per_gpu * $gpus_per_node))
./manage-node-pool.sh create \
        -n tritonserver-pool \
        -c ${cluster} \
        -p ${project} \
        -z ${zone} \
        -g ${gpus_per_node} \
        -N ${nodes} \
        -v ${vcpus_per_node} || echo "Triton node pool already created" 2>/dev/null

# if we asked for the full number of gpus in a deployment,
# we won't have enough CPU, so take one off
vcpus_per_deployment=$(($vcpus_per_gpu * $gpus_per_deployment))
if [[ $vcpus_per_deployment == $vcpus_per_node ]]; then
    vcpus_per_deployment=$(($vcpus_per_deployment - 1))
fi

python format_yaml.py apps/triton-server/deploy.yaml \
    --gpus ${gpus_per_deployment} \
    --vcpus ${vcpus_per_deployment} \
    --repo ${bucket} \
    --name ${name} | kubectl apply -f -
kubectl rollout status deployment/${name}

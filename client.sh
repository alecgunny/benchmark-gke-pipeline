#! /bin/bash -e

while getopts "c:p:z:b:G:g:v:N:r:i:q:l:h" opt; do
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
        G )
            gpus_per_node=${OPTARG}
            ;;
        g )
            gpus_per_deployment=${OPTARG}
            ;;
        v )
            vcpus_per_gpu=${OPTARG}
            ;;
        N )
            nodes=${OPTARG}
            ;;
        r )
            rate=${OPTARG}
            ;;
        i )
            iterations=${OPTARG}
            ;;
        q )
            queue=${OPTARG}
            ;;
        l )
            latency=${OPTARG}
            ;;
        h )
            echo "Create a GPU node pool and deploy Triton server"
            echo "deployments and load balancers onto it, then run"
            echo "multi-client benchmarking scripts against this."
            echo "Creates a separate deployment and load balancer"
            echo "for each node so that clients can point to"
            echo "ip addresses dedicated to their inference stream."
            echo ""
            echo "Options:"
            echo "--------"
            echo "    -c: cluster name for deploying Triton server"
            echo "    -p: project name for deploying Triton server"
            echo "    -z: zone for deploying Triton server"
            echo "    -b: GCP bucket where Triton model repository is hosted"
            echo "    -G: Desired number of GPUs per node in server node pool"
            echo "    -g: Desired number of GPUs per deployment. Must be less than or equal to -G"
            echo "    -v: Desired number of vCPUs per GPU"
            echo "    -N: Number of server nodes (and client instances) to leverage"
            echo "    -r: Rate at which to generate requests from each client"
            echo "    -i: Number of iterations over which to run client benchmarking"
            echo "    -q: Maximum allowable queuing time for any model in microseconds"
            echo "    -l: Maximum allowable end-to-end latency in seconds"
            exit 0
            ;;
        \? )
            echo "Unrecognized argument ${opt}"
            exit 1
    esac
done
shift $((OPTIND -1))

: ${cluster:?Must specify cluster name to deploy server on}
: ${project:?Must specify GCP project for cluster}
: ${zone:?Must specify zone for cluster}
: ${bucket:?Must specify GCS bucket hosting model repo}
: ${gpus_per_node:?Must specify number of GPUs to leverage per node in node pool}
: ${gpus_per_deployment:?Must specify number of GPUs to leverage in server deployments}
: ${vcpus_per_gpu:?Must specify number of vCPUs to leverage per GPU}

nodes=${nodes:-1}
for i in $(seq $nodes); do
    if [[ $nodes == 1 ]]; then
        name="tritonserver"
    else
        name="tritonserver-${i}"
    fi

    ./start-server.sh \
        -c ${cluster} \
        -p ${project} \
        -z ${zone} \
        -b ${bucket} \
        -G ${gpus_per_node} \
        -g ${gpus_per_deployment} \
        -v ${vcpus_per_gpu} \
        -N ${nodes} \
        -n ${name}
done

queue=${queue:-100000}
latency=${latency:-1}
for i in $(seq $nodes); do
    if [[ $nodes == 1 ]]; then
        name="tritonserver"
    else
        name="tritonserver-${i}"
    fi

    ip=$(kubectl get service $name -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [[ $i == $nodes ]]; then
        python client.py \
            --url $ip:8001 \
            --model-name gwe2e \
            --model-version 1 \
            --generation-rate ${rate} \
            --num-iterations ${iterations} \
            --warm-up 10 \
            --file-prefix "node-${i}" \
            --queue-threshold-us ${queue} \
            --latency-threshold ${latency} 2>&1 | tee node-${i}_output.log
    else
        python client.py \
            --url $ip:8001 \
            --model-name gwe2e \
            --model-version 1 \
            --generation-rate ${rate} \
            --num-iterations ${iterations} \
            --warm-up 10 \
            --file-prefix "node-${i}" \
            --queue-threshold-us ${queue} \
            --latency-threshold ${latency} &> node-${i}_output.log
    fi
done

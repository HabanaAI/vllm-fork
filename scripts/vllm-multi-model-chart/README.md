# Deploy vLLM with Multiple Models using Intel® Gaudi® AI accelerators with Kubernetes

This example demonstrates using vLLM to serve multiple models with a single Intel Gaudi HPU from a Kubernetes cluster.

## Prerequisites

To run this example you will need:
* Docker to build and push the docker image
* A Kubernetes cluster with Intel® Gaudi® Al accelerators (see the
  [Kubernetes Installation](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Kubernetes_Installation/index.html)
  instructions for Intel Gaudi or the [Intel Kubernetes Service Guide](https://console.cloud.intel.com/docs/guides/k8s_guide.html)
  for using a Kubernetes cluster from Intel® Tiber™ AI Cloud)
* [`kubectl`](https://kubernetes.io/docs/tasks/tools/#kubectl) installed and configured
* [Helm](https://helm.sh/docs/intro/quickstart/) installed and configured

## Docker Image

Before deploying vLLM to Kubernetes, you will first need to build and push a vLLM docker image for HPU. To do this,
clone the `multi_model` branch from the [HabanaAI fork of vLLM](https://github.com/HabanaAI/vllm-fork/tree/multi_model),
and build the image using the `Dockerfile.hpu` file. Push the image to a registry that will be accessible by your
Kubernetes cluster nodes.

```
# Clone the multi_model branch of the HabanaAI fork of vLLM
git clone https://github.com/HabanaAI/vllm-fork.git --branch multi_model --single-branch --depth 1
cd vllm-fork

# Build the docker image
$ docker build -f Dockerfile.hpu -t vllm-hpu-env .

# Tag and push the image to a container registry
docker tag vllm-hpu-env <registry/image:tag>
docker push <registry/image:tag>
```

### Deploy vLLM using Kubernetes

After the docker image has been built and pushed, use the [Helm](https://helm.sh) chart to deploy vLLM to your cluster.

The Helm chart includes:
* Secret used for storing a [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens)
* [Persistent volume claim (PVC)](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) for the [Hugging Face Hub model cache](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache) directory
* Deployment to run the vLLM model server
* Service to expose the deployment

1. Before installing the Helm chart, update the [`values.yaml` file](values.yaml) to configure your job. Important
   values that you will need to set are:
   * Set `image.repository` and `image.tag` based on the docker image that you have built and pushed.
   * Set`secret.hfToken` with your Hugging Face token, if you are using gated models.
   * The `args` section has the command used launch vLLM. In those args, specify the name of the models to serve.
   * Set your Kubernetes storage class name in the `storage.storageClassName` field.
   * The `resources` section defines the resource limits and requests for the deployment.

2. After configuring the [`values.yaml`](values.yaml) file, the Helm chart can be deployed to the cluster:

   ```
   cd scripts/vllm-multi-model-chart
   helm install -f values.yaml gaudi-multi-model .
   ```

3. Monitor the job by check the pod status, and viewing the logs. It will take a few minutes for the model weights to
   download. Note that if the same PVC is reused, subsequent runs can use cached models files.

   ```
   kubectl get pods

   kubectl logs <gaudi-multi-model-deployment pod name> -f
   ```

   When it's ready you will see a message about Uvicorn running on port 8000.
4. If you don't have an external IP for the node running the vLLM deployment, port forward in order to access the
   vLLM service from your local machine. In a separate terminal port forward:

   ```
   kubectl port-forward svc/gaudi-multi-model-service 8000:80
   ```

5. Now that the service is up and running, you can use cURL commands to test the deployment. For example:
   List the running models:

   ```
   curl http://localhost:8000/v1/models
   ```

   Query the served models with prompt inputs (change the `model`, depending on the names of the models that you
     served - by default, the values file uses `meta-llama/Llama-3.1-8B-Instruct` and `mistralai/Mistral-7B-Instruct-v0.3`):
  
   ```
   curl http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
   -d '{
         "model": "meta-llama/Llama-3.1-8B-Instruct",
         "prompt": "San Francisco is a",
         "max_tokens": 20,
         "temperature": 0
         }'
   ```

   Swap out the models being served:

   ```
   curl http://localhost:8000/v1/update_models \
      -H "Content-Type: application/json" \
      -d '{"models": [{"id":"mistralai/Mistral-7B-Instruct-v0.3"},{"id":"Qwen/Qwen2.5-7B-Instruct"}]}'
   ```

We have demonstrated serving multiple models using a single HPU from a Kubernetes cluster using vLLM. When you are done,
the vLLM deployment and service can be shut down to free resources using:

```
kubectl delete deployment gaudi-multi-model-deployment
kubectl delete service gaudi-multi-model-service
```

This leaves the PVC and secret resources for use with future deployments
(using `helm upgrade -f values.yaml gaudi-multi-model .`).

If you would like to delete all Helm chart resources, uninstall the chart:

```
helm uninstall gaudi-multi-model
```

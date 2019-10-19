import os
import yaml
from kubernetes import client, config
from kubernetes.utils.create_from_yaml import create_from_yaml_single_item
import subprocess


def create_job(yaml_file, env_vars):
    with open(yaml_file) as f:
        yaml_str = f.read()
    env_vars = {v: str(env_vars[v]) for v in env_vars}
    os.environ.update(env_vars)
    yaml_str = os.path.expandvars(yaml_str)
    job_spec = yaml.load(yaml_str, Loader=yaml.FullLoader)
    k8s_client = client.api_client.ApiClient()
    create_from_yaml_single_item(k8s_client, job_spec)


def load_config():
    config.load_kube_config()


def get_pod_with_name(name_filters):
    k8s_client = client.CoreV1Api()
    pod_list = k8s_client.list_pod_for_all_namespaces()
    pods = [
        pod for pod in pod_list.items
        if pod.metadata.labels and 'job-name' in pod.metadata.labels
    ]

    pods = [
        pod for pod in pods if all(
            [name in pod.metadata.labels['job-name'] for name in name_filters])
    ]

    return pods[0]


def copy_to_pod(pod, source_file, dest_dir):
    pm = pod.metadata
    subprocess.check_call([
        'kubectl',
        'cp',
        source_file,
        pm.namespace + '/' + pm.name + ':' + dest_dir,
    ])

import openvino as ov
import os
import re
from transformers import AutoConfig

def rileva_device():
    core = ov.Core()
    devices = core.available_devices

    model_device_name_GPU = []
    gpu_device_id = []
    model_device_name_CPU = "CPU"

    for device in devices : 
        full_name = core.get_property(device, "FULL_DEVICE_NAME")

        if "GPU" in full_name :
            model_device_name_GPU.append(full_name)
            gpu_device_id.append(device)
        elif "CPU" in full_name :
            model_device_name_CPU = full_name
        
    return model_device_name_GPU, model_device_name_CPU, gpu_device_id

def compose_path(generic_dir, file):
    return os.path.join(generic_dir, file)

def print_all_contents(generic_dir):
    contents = os.listdir(generic_dir)
    for i, content in enumerate(contents):
        to_check = check_file(generic_dir, content)
        if to_check:
            print(f"{i} - [FILE] {content}")
        else:
            print(f"{i} - [FOLDER] {content}")

def check_file(base_dir, generic_dir) -> bool:
    to_check = os.path.join(base_dir, generic_dir)
    if os.path.isfile(to_check):
        return True
    else:
        return False

def check_folder(generic_dir) -> bool:
    contents = os.listdir(generic_dir)
    for f in contents:
        if os.path.isdir(os.path.join(generic_dir, f)):
            return True
    return False

def lettura_file_config(model_name):
    config = AutoConfig.from_pretrained(model_name)

    print(config.to_dict())
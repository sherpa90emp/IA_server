import openvino as ov
import os

def rileva_device():
    core = ov.Core()
    devices = core.available_devices

    model_device_name_GPU = "Non trovata"
    model_device_name_CPU = "Non trovata"
    target_device = "CPU"

    for device in devices : 
        full_name = core.get_property(device, "FULL_DEVICE_NAME")

        if "GPU" in full_name :
            model_device_name_GPU = full_name
            target_device = "GPU"
        elif "CPU" in full_name :
            model_device_name_CPU = full_name

    return model_device_name_GPU, model_device_name_CPU, target_device

def compose_path(dir, file):
    return os.path.join(dir, file)
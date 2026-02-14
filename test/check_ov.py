import openvino as ov
core = ov.Core()
devices = core.available_devices

print(f"Dispositivi trovati: {devices}")

for device in devices:
    full_name = core.get_property(device, "FULL_DEVICE_NAME")
    print(f"Dispositivo: {device} -> {full_name}")
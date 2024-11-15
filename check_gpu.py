import torch

def check_cuda_devices():
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {num_gpus}")

    for i in range(num_gpus):
        device_name = torch.cuda.get_device_name(i)
        device_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3 
        print(f"Device {i}: {device_name} with {device_memory:.2f} GB memory")

if __name__ == "__main__":
    check_cuda_devices()
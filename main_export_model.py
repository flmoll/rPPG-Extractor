import argparse
import torch
import torch.onnx
import onnxruntime
import numpy as np
import os

from tqdm import tqdm
from main import get_data_loader, get_config, add_args

from neural_methods.model.model_wrappers import PhysNet_Wrapper
from neural_methods.model.HRClassifierUncertainty import HRClassifierUncertainty

from torch.profiler import profile, record_function, ProfilerActivity

if __name__ == "__main__":

    # Paths
    onnx_model_path = "model.onnx"
    ort_model_path = "model.ort"
    weights_path = ""   # Insert path here
    device = 'cpu'

    # Step 1: Load and prepare model
    model = PhysNet_Wrapper(frames=160, in_channels=3) # change to your model
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    dummy_input = torch.randn(1, 2, 160)


    traced = torch.jit.trace(model, dummy_input)
    traced.save("model.pt")
    print("✅ Saved traced model: model.pt")


    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    config = get_config(args)
    calibration_loader = get_data_loader(config.VALID, device=device, batch_size=1)

    # Attach observers
    model.qconfig = torch.quantization.QConfig(
        activation=torch.quantization.default_observer,
        weight=torch.quantization.default_weight_observer  # uses per-tensor quant
    )
    torch.quantization.prepare(model, inplace=True)
    model = model.to(device)

    count = 0

    for batch, identifier, idx in tqdm(calibration_loader):
        inputs = batch['data']
        inputs = inputs.to(device)
        # Forward pass to collect statistics
        model(inputs)
        count += 1

        if count >= 10:
            break

    model_int8 = torch.quantization.convert(model.eval(), inplace=False)
    model_int8_traced = torch.jit.trace(model_int8, dummy_input)
    model_int8_traced.save("model_int8.pt")
    print("✅ Saved quantized traced model: model_int8.pt")

    print(model_int8_traced)
    
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            out = model_int8(dummy_input)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # Step 2: Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=18,  # ✅ use opset 11–13 for mobile compatibility
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f"✅ Exported to ONNX: {onnx_model_path}")

    # Step 3: Convert to ORT format for mobile
    os.system(f"python -m onnxruntime.tools.convert_onnx_models_to_ort {onnx_model_path} --output {ort_model_path}")


    print(f"✅ Converted to ORT: {ort_model_path}")

    # Run inference with ONNX Runtime (ORT format model)
    so = onnxruntime.SessionOptions()
    ort_session = onnxruntime.InferenceSession(ort_model_path + "/model.ort", so, providers=['CPUExecutionProvider'])

    ort_inputs = {"input": dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_output = ort_outs[0]

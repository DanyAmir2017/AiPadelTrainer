"""
Convert YOLOv8n model to Hailo-8 compatible ONNX format.
Replaces unsupported activations (SiLU) with ReLU for Hailo NPU compatibility.
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
import onnx
from pathlib import Path

def replace_activations(model, old_activation=nn.SiLU, new_activation=nn.ReLU):
    """
    Recursively replace activation functions in the model.
    
    Args:
        model: PyTorch model
        old_activation: Activation to replace (default: SiLU)
        new_activation: New activation (default: ReLU)
    """
    count = 0
    for child_name, child in model.named_children():
        if isinstance(child, old_activation):
            setattr(model, child_name, new_activation(inplace=True))
            count += 1
            print(f"  Replaced {old_activation.__name__} → {new_activation.__name__} in layer: {child_name}")
        else:
            # Recursively process child modules
            count += replace_activations(child, old_activation, new_activation)
    return count

def convert_model_to_hailo_compatible(
    model_path,
    output_path="best_ball_nano_hailo.onnx",
    img_size=640,
    opset=11
):
    """
    Convert YOLOv8n model to Hailo-compatible ONNX format.
    
    Args:
        model_path: Path to trained YOLOv8n .pt model
        output_path: Path for output ONNX model
        img_size: Input image size (default: 640)
        opset: ONNX opset version (default: 11 for Hailo)
    
    Returns:
        Path to exported ONNX model
    """
    print("="*60)
    print("YOLOv8n to Hailo-8 Compatible ONNX Converter")
    print("="*60)
    
    # 1. Load the trained model
    print(f"\n[1/5] Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Get the PyTorch model
    pt_model = model.model
    print(f"  Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in pt_model.parameters()):,}")
    
    # 2. Replace SiLU activations with ReLU
    print(f"\n[2/5] Replacing unsupported activations...")
    print(f"  Searching for SiLU activations...")
    count = replace_activations(pt_model, nn.SiLU, nn.ReLU)
    print(f"  ✓ Replaced {count} SiLU → ReLU activations")
    
    # Also check for other potentially unsupported activations
    # Hailo generally supports: ReLU, LeakyReLU, but not SiLU, Mish, etc.
    
    # 3. Set model to evaluation mode
    print(f"\n[3/5] Preparing model for export...")
    pt_model.eval()
    print(f"  Model set to evaluation mode")
    
    # 4. Export to ONNX with Hailo-compatible settings
    print(f"\n[4/5] Exporting to ONNX...")
    print(f"  Output: {output_path}")
    print(f"  Image size: {img_size}")
    print(f"  Opset: {opset}")
    print(f"  Dynamic shapes: False (static for Hailo)")
    
    try:
        # Use Ultralytics export with specific settings
        export_path = model.export(
            format='onnx',
            imgsz=img_size,
            opset=opset,
            dynamic=False,  # Static shapes required for Hailo
            simplify=True,  # Simplify ONNX graph
            half=False      # FP32 for compatibility
        )
        
        # Rename to desired output name
        export_path_obj = Path(export_path)
        output_path_obj = Path(output_path)
        if export_path_obj != output_path_obj:
            import shutil
            shutil.copy(export_path, output_path)
            print(f"  ✓ Copied to: {output_path}")
        
        print(f"  ✓ ONNX export successful!")
        
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        print("\n  Trying alternative export method...")
        
        # Alternative: Manual ONNX export
        dummy_input = torch.randn(1, 3, img_size, img_size)
        torch.onnx.export(
            pt_model,
            dummy_input,
            output_path,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes=None,  # Static shapes
            opset_version=opset,
            do_constant_folding=True
        )
        print(f"  ✓ Manual ONNX export successful!")
    
    # 5. Verify ONNX model
    print(f"\n[5/5] Verifying ONNX model...")
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"  ✓ ONNX model is valid!")
        
        # Print model info
        print(f"\n  Model Information:")
        print(f"    - IR Version: {onnx_model.ir_version}")
        print(f"    - Opset: {onnx_model.opset_import[0].version}")
        print(f"    - Inputs: {[inp.name for inp in onnx_model.graph.input]}")
        print(f"    - Outputs: {[out.name for out in onnx_model.graph.output]}")
        
        # Check for unsupported ops
        print(f"\n  Checking for Hailo-unsupported operations...")
        unsupported_ops = []
        for node in onnx_model.graph.node:
            if node.op_type in ['Sigmoid', 'Swish', 'Mish', 'HardSwish']:
                unsupported_ops.append(node.op_type)
        
        if unsupported_ops:
            print(f"  ⚠ Warning: Found potentially unsupported ops: {set(unsupported_ops)}")
        else:
            print(f"  ✓ No known unsupported ops detected!")
        
    except Exception as e:
        print(f"  ✗ ONNX validation failed: {e}")
        return None
    
    print("\n" + "="*60)
    print("✓ Conversion Complete!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Transfer to Raspberry Pi:")
    print(f"   scp {output_path} padel-pi:~/")
    print(f"\n2. On Pi, parse to HAR:")
    print(f"   hailo parser onnx {Path(output_path).name}")
    print(f"\n3. Compile to HEF:")
    print(f"   hailo compiler {Path(output_path).stem}.har --hw-arch hailo8")
    print(f"\n4. Test inference:")
    print(f"   hailortcli run {Path(output_path).stem}.hef")
    print("="*60)
    
    return output_path

if __name__ == "__main__":
    import sys
    
    # Default model path (can be overridden via command line)
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Try to find the trained model
        possible_paths = [
            "runs/detect/runs/detect/ball_nano3/weights/best.pt",
            "models/best_ball_nano.pt",
            "best.pt"
        ]
        
        model_path = None
        for path in possible_paths:
            if Path(path).exists():
                model_path = path
                break
        
        if model_path is None:
            print("Error: Could not find trained model!")
            print("Usage: python convert_to_hailo.py <path_to_best.pt>")
            sys.exit(1)
    
    # Output path
    output_path = "models/onnx/best_ball_nano_hailo.onnx"
    
    # Convert
    result = convert_model_to_hailo_compatible(
        model_path=model_path,
        output_path=output_path,
        img_size=640,
        opset=11
    )
    
    if result:
        print(f"\n✓ Success! Hailo-compatible model saved to: {result}")
    else:
        print(f"\n✗ Conversion failed. Check errors above.")
        sys.exit(1)

from ultralytics import YOLO
import os

models_dir = "models"
model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]

print("\n" + "="*60)
print("MODEL ANALYSIS")
print("="*60)

variant_map = {
    (0.33, 0.25): "YOLOv8n (nano)",
    (0.33, 0.50): "YOLOv8s (small)",
    (0.67, 0.75): "YOLOv8m (medium)",
    (1.0, 1.0): "YOLOv8l (large)",
    (1.0, 1.25): "YOLOv8x (extra-large)"
}

for model_file in sorted(model_files):
    model_path = os.path.join(models_dir, model_file)
    print(f"\n{model_file}:")
    
    try:
        model = YOLO(model_path)
        depth = model.model.yaml['depth_multiple']
        width = model.model.yaml['width_multiple']
        params = sum(p.numel() for p in model.model.parameters())
        
        variant = variant_map.get((depth, width), f"Unknown (d={depth}, w={width})")
        
        print(f"  Variant: {variant}")
        print(f"  Parameters: {params:,}")
        print(f"  Size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
        
        # Edge deployment suitability
        if params < 5_000_000:
            print(f"  Edge Status: ✓ Excellent for Pi/Jetson")
        elif params < 15_000_000:
            print(f"  Edge Status: ~ Moderate for Jetson")
        elif params < 30_000_000:
            print(f"  Edge Status: ⚠ Heavy, needs powerful edge device")
        else:
            print(f"  Edge Status: ✗ Too heavy for edge deployment")
            
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "="*60)
print("\nRECOMMENDATION:")
print("For Raspberry Pi / Jetson Nano edge deployment:")
print("  → Use YOLOv8n (nano) with ~3M parameters")
print("  → Current YOLOv8x models are too heavy (68M params)")
print("  → Consider retraining with 'yolov8n.pt' as base model")
print("="*60)

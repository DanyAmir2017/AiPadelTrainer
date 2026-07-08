"""
Generate System Architecture Diagram for Padel Trainer
=======================================================

Creates a system architecture flowchart showing the complete pipeline
with minimalistic academic style: white background, blue/dark-gray blocks,
clean arrows, and modular layout.

Usage:
    python generate_system_architecture.py --output PaperLatex/figures/system_architecture.png

Author: Padel Trainer
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path


def create_system_architecture(output_path):
    """
    Create system architecture diagram with minimalistic academic style.
    
    Args:
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300, facecolor='white')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define color palette - minimalistic
    color_input = '#1E3A8A'       # Dark blue
    color_detection = '#3B82F6'   # Bright blue
    color_tracking = '#60A5FA'    # Light blue
    color_processing = '#0D47A1'  # Navy
    color_output = '#1565C0'      # Deep blue
    text_color = 'white'
    edge_color = '#0F172A'        # Very dark gray
    
    # Box dimensions
    box_w = 1.8
    box_h = 0.6
    
    # Helper function to create styled boxes
    def create_box(x, y, width, height, text, color, fontsize=9, bold=False):
        """Create minimalistic box."""
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.05", 
            edgecolor=edge_color, 
            facecolor=color,
            linewidth=1.5,
            zorder=2
        )
        ax.add_patch(box)
        
        weight = 'bold' if bold else 'normal'
        ax.text(x, y, text, 
               horizontalalignment='center',
               verticalalignment='center',
               fontsize=fontsize,
               fontweight=weight,
               color=text_color,
               zorder=3)
    
    # Helper function for arrows
    def create_arrow(x1, y1, x2, y2, curve=0.0, width=1.5):
        """Create clean arrow."""
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->,head_width=0.25,head_length=0.25',
            color=edge_color,
            linewidth=width,
            connectionstyle=f"arc3,rad={curve}",
            zorder=1
        )
        ax.add_patch(arrow)
    
    # Row 1: Input
    y_input = 9
    create_box(2, y_input, 2.5, 0.8, 'Video Input\n(MP4)', color_input, fontsize=10, bold=True)
    create_arrow(2, y_input - 0.4, 2, 8.1)
    
    # Row 2: Frame Extraction
    y_frame = 8
    create_box(2, y_frame, 2, 0.6, 'Frame\nExtraction', color_detection, fontsize=9)
    create_arrow(2, y_frame - 0.3, 2, 7.2)
    
    # Row 3: Detection Pipeline (parallel)
    y_detect = 6.5
    
    # Ball Detection
    create_box(0.8, y_detect, 1.6, 0.6, 'Ball\nDetection', color_detection)
    create_arrow(1.5, y_frame - 0.3, 0.8, y_detect + 0.3)
    
    # Player Detection
    create_box(2.6, y_detect, 1.6, 0.6, 'Player\nDetection', color_detection)
    create_arrow(2, y_frame - 0.3, 2.6, y_detect + 0.3)
    
    # Court Detection
    create_box(4.4, y_detect, 1.6, 0.6, 'Court\nDetection', color_detection)
    create_arrow(2.5, y_frame - 0.3, 4.4, y_detect + 0.3)
    
    # Merge arrow
    create_arrow(0.8, y_detect - 0.3, 2, 5.8)
    create_arrow(2.6, y_detect - 0.3, 2, 5.8)
    create_arrow(4.4, y_detect - 0.3, 2, 5.8)
    
    # Row 4: Fusion/Merge
    y_fusion = 5.5
    create_box(2, y_fusion, 2, 0.6, 'Detection\nFusion', color_processing, fontsize=9)
    create_arrow(2, y_fusion - 0.3, 2, 4.6)
    
    # Row 5: Tracking Pipeline (parallel)
    y_track = 4.2
    
    # YOLO Tracking
    create_box(0.8, y_track, 1.6, 0.6, 'YOLO\nTracking', color_tracking)
    create_arrow(1.5, y_fusion - 0.3, 0.8, y_track + 0.3)
    
    # Optical Flow
    create_box(2.6, y_track, 1.6, 0.6, 'Optical\nFlow', color_tracking)
    create_arrow(2, y_fusion - 0.3, 2.6, y_track + 0.3)
    
    # Kalman Filter
    create_box(4.4, y_track, 1.6, 0.6, 'Kalman\nFilter', color_tracking)
    create_arrow(2.5, y_fusion - 0.3, 4.4, y_track + 0.3)
    
    # Merge tracking
    create_arrow(0.8, y_track - 0.3, 2, 3.4)
    create_arrow(2.6, y_track - 0.3, 2, 3.4)
    create_arrow(4.4, y_track - 0.3, 2, 3.4)
    
    # Row 6: Track Fusion
    y_track_fusion = 3.0
    create_box(2, y_track_fusion, 2.2, 0.6, 'Track\nFusion', color_processing, fontsize=9)
    create_arrow(2, y_track_fusion - 0.3, 2, 2.2)
    
    # Row 7: Event Detection (right branch)
    y_events = 4.2
    create_box(6.5, y_events, 1.8, 0.6, 'Shot\nDetection', color_detection)
    create_arrow(3.1, y_fusion - 0.1, 5.6, y_events + 0.1, curve=0.3)
    
    create_box(8.5, y_events, 1.8, 0.6, 'Contact\nDetection', color_detection)
    create_arrow(7.4, y_events, 7.6, y_events)
    
    # Merge events down
    create_arrow(6.5, y_events - 0.3, 5, 2.5, curve=-0.2)
    create_arrow(8.5, y_events - 0.3, 5, 2.5, curve=0.2)
    
    # Row 8: Analysis/Metrics
    y_analysis = 1.5
    create_box(2, y_analysis, 2, 0.6, 'Trajectory\nAnalysis', color_processing, fontsize=9)
    create_arrow(2, y_track_fusion - 0.3, 2, y_analysis + 0.3)
    
    # Row 9: Output (parallel)
    y_output = 0.5
    
    # Annotated Video
    create_box(0.6, y_output, 1.5, 0.5, 'Annotated\nVideo', color_output, fontsize=8)
    create_arrow(1.2, y_analysis - 0.3, 0.6, y_output + 0.25)
    
    # Metrics CSV
    create_box(2.4, y_output, 1.5, 0.5, 'Metrics\nCSV', color_output, fontsize=8)
    create_arrow(2, y_analysis - 0.3, 2.4, y_output + 0.25)
    
    # Trajectory CSV
    create_box(4.2, y_output, 1.5, 0.5, 'Trajectory\nCSV', color_output, fontsize=8)
    create_arrow(2.8, y_analysis - 0.3, 4.2, y_output + 0.25)
    
    # ============ RIGHT SIDE: Edge Deployment ============
    
    # Title for edge pipeline
    ax.text(10.5, 9.5, 'Edge Deployment Pipeline',
           fontsize=11, fontweight='bold', color=edge_color, ha='center')
    
    # Input
    create_box(10.5, 8.8, 2, 0.6, 'RPi 5 +\nHailo-8', color_input, fontsize=9, bold=True)
    create_arrow(10.5, 8.5, 10.5, 7.9)
    
    # ONNX Models
    create_box(9, 7.2, 1.8, 0.6, 'ONNX\nBall', color_detection, fontsize=8)
    create_box(10.5, 7.2, 1.8, 0.6, 'ONNX\nPlayers', color_detection, fontsize=8)
    create_box(12, 7.2, 1.8, 0.6, 'ONNX\nCourt', color_detection, fontsize=8)
    
    create_arrow(10.5, 7.5, 9, 7.5)
    create_arrow(10.5, 7.5, 10.5, 7.5)
    create_arrow(10.5, 7.5, 12, 7.5)
    
    # Inference
    create_box(10.5, 6.2, 2.5, 0.6, 'Hailo\nInference', color_detection, fontsize=9)
    create_arrow(9, 6.9, 9.6, 6.5)
    create_arrow(10.5, 6.9, 10.5, 6.5)
    create_arrow(12, 6.9, 11.4, 6.5)
    
    create_arrow(10.5, 5.9, 10.5, 5.2)
    
    # Lightweight Tracking
    create_box(10.5, 4.8, 2.2, 0.6, 'Lightweight\nTracking', color_tracking, fontsize=9)
    create_arrow(10.5, 4.5, 10.5, 3.8)
    
    # Edge Output
    create_box(10.5, 3.4, 2.2, 0.6, 'Real-time\nOutput', color_output, fontsize=9)
    
    # Connection between pipelines
    ax.plot([5.1, 8.2], [2, 2], 'k--', linewidth=1, alpha=0.3, zorder=0)
    ax.text(6.6, 2.15, 'Integration', fontsize=8, style='italic', 
           color='gray', ha='center', alpha=0.7)
    
    # Legend
    legend_y = 0.5
    ax.text(7.2, legend_y + 0.2, 'Processing Stages:', fontsize=9, fontweight='bold', color=edge_color)
    
    legend_items = [
        (color_input, 'Input/Hardware'),
        (color_detection, 'Detection'),
        (color_tracking, 'Tracking'),
        (color_processing, 'Fusion/Processing'),
        (color_output, 'Output')
    ]
    
    for i, (color, label) in enumerate(legend_items):
        x = 7.2 + (i % 3) * 2.0
        y = legend_y - (i // 3) * 0.35
        
        rect = patches.Rectangle((x, y - 0.08), 0.15, 0.15, 
                                 facecolor=color, edgecolor=edge_color, linewidth=0.5)
        ax.add_patch(rect)
        ax.text(x + 0.25, y + 0.02, label, fontsize=7.5, va='center', color=edge_color)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png', facecolor='white')
    print(f"✓ Figure saved: {output_path}")
    
    # PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', format='pdf', facecolor='white')
    print(f"✓ PDF version saved: {pdf_path}")
    
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate system architecture diagram')
    parser.add_argument('--output', type=str, 
                       default='PaperLatex/figures/system_architecture.png',
                       help='Output figure path')
    
    args = parser.parse_args()
    
    print(f"Generating system architecture diagram...")
    print(f"  Style: Minimalistic academic")
    print(f"  Output: {args.output}")
    
    create_system_architecture(args.output)
    print("\n✓ System architecture diagram complete!")


if __name__ == '__main__':
    main()

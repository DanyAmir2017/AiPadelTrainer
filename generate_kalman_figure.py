"""
Generate Kalman Filter Prediction Model Diagram for Thesis
===========================================================

Creates a flowchart illustrating the Kalman filter prediction and correction cycle,
showing the iterative process of state estimation in ball tracking.

Usage:
    python generate_kalman_figure.py --output PaperLatex/figures/kalman_filter_fig.png

Author: Padel Trainer
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path


def create_kalman_diagram(output_path):
    """
    Create Kalman filter prediction model diagram.
    
    Args:
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 12), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define colors
    color_state = '#E8F4F8'      # Light blue for state
    color_predict = '#FFF4E6'     # Light orange for prediction
    color_measure = '#E8F5E9'     # Light green for measurement
    color_correct = '#FCE4EC'     # Light pink for correction
    
    edge_color = '#333333'
    text_color = '#1a1a1a'
    
    # Box dimensions
    box_width = 3.5
    box_height = 0.8
    
    # Helper function to create styled boxes
    def create_box(ax, x, y, width, height, text, color, fontsize=11, fontweight='normal'):
        """Create a styled box with text."""
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.1", 
            edgecolor=edge_color, 
            facecolor=color,
            linewidth=2.5,
            zorder=2
        )
        ax.add_patch(box)
        
        ax.text(x, y, text, 
               horizontalalignment='center',
               verticalalignment='center',
               fontsize=fontsize,
               fontweight=fontweight,
               color=text_color,
               zorder=3)
    
    # Helper function to create arrows
    def create_arrow(ax, x1, y1, x2, y2, label='', curve=0.0):
        """Create styled arrow between boxes."""
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->,head_width=0.4,head_length=0.4',
            color=edge_color,
            linewidth=2.5,
            connectionstyle=f"arc3,rad={curve}",
            zorder=1
        )
        ax.add_patch(arrow)
        
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.3, mid_y + 0.2, label,
                   fontsize=9, style='italic', color=text_color, zorder=3)
    
    # Layout positions
    y_positions = [13, 11.5, 10, 8.5, 7, 5.5, 4, 2.5, 1]
    x_center = 5
    
    # Title
    ax.text(x_center, 13.5, 'Kalman Filter Prediction & Correction Cycle',
           fontsize=14, fontweight='bold', horizontalalignment='center', 
           color=text_color, zorder=3)
    
    # Step 1: Previous State
    create_box(ax, x_center, y_positions[0], box_width, box_height,
              'Previous State\n$\\mathbf{x}_{k-1}$', color_state, fontweight='bold')
    
    # Arrow down
    create_arrow(ax, x_center, y_positions[0] - box_height/2,
                x_center, y_positions[1] + box_height/2)
    
    # Step 2: Prediction Step
    create_box(ax, x_center, y_positions[1], box_width, box_height,
              'Prediction Step\n$\\mathbf{x}_{k}^{-} = \\mathbf{A} \\mathbf{x}_{k-1}^{+}$', 
              color_predict, fontsize=10)
    
    # Arrow down with annotation
    create_arrow(ax, x_center, y_positions[1] - box_height/2,
                x_center, y_positions[2] + box_height/2)
    
    # Step 3: Predicted Position
    create_box(ax, x_center, y_positions[2], box_width, box_height,
              'Predicted Position\n$\\mathbf{x}_{k}^{-}$', color_state, fontweight='bold')
    
    # Arrow down
    create_arrow(ax, x_center, y_positions[2] - box_height/2,
                x_center, y_positions[3] + box_height/2)
    
    # Step 4: Measurement
    create_box(ax, x_center, y_positions[3], box_width, box_height,
              'Measurement Update\n$\\mathbf{z}_k$ (YOLO/OptFlow)', color_measure)
    
    # Arrow down
    create_arrow(ax, x_center, y_positions[3] - box_height/2,
                x_center, y_positions[4] + box_height/2)
    
    # Step 5: Correction Step
    create_box(ax, x_center, y_positions[4], box_width, box_height,
              'Correction Step\n$\\mathbf{x}_{k}^{+} = \\mathbf{x}_{k}^{-} + \\mathbf{K}_k(\\mathbf{z}_k - \\mathbf{x}_{k}^{-})$', 
              color_correct, fontsize=9)
    
    # Arrow down
    create_arrow(ax, x_center, y_positions[4] - box_height/2,
                x_center, y_positions[5] + box_height/2)
    
    # Step 6: Corrected State
    create_box(ax, x_center, y_positions[5], box_width, box_height,
              'Corrected State\n$\\mathbf{x}_{k}^{+}$', color_state, fontweight='bold')
    
    # Covariance Update box (side panel)
    covar_x = 8.5
    create_box(ax, covar_x, y_positions[4], 2.5, 1.5,
              'Covariance\nUpdate\n$\\mathbf{P}_{k}^{+}$', 
              '#F3E5F5', fontsize=9)
    
    # Arrow from correction to covariance
    create_arrow(ax, x_center + box_width/2, y_positions[4],
                covar_x - 1.25, y_positions[4], curve=0.3)
    
    # Loop arrow (feedback)
    loop_x1 = x_center + box_width/2 + 0.3
    loop_y1 = y_positions[5] - box_height/2
    loop_x2 = x_center + box_width/2 + 0.3
    loop_y2 = y_positions[0] + box_height/2
    
    curved_arrow = FancyArrowPatch(
        (loop_x1, loop_y1), (loop_x2, loop_y2),
        arrowstyle='->,head_width=0.4,head_length=0.4',
        color='#D32F2F',
        linewidth=2.5,
        connectionstyle="arc3,rad=-0.8",
        zorder=1
    )
    ax.add_patch(curved_arrow)
    
    # Feedback label
    ax.text(loop_x1 + 1.5, (loop_y1 + loop_y2)/2, 'Next Iteration',
           fontsize=9, style='italic', color='#D32F2F', zorder=3)
    
    # Add math notation box
    math_box_y = 0.5
    ax.text(x_center, math_box_y - 0.3, 
           'Key: $\\mathbf{A}$ = State transition matrix | $\\mathbf{K}_k$ = Kalman gain | $\\mathbf{P}_k$ = Covariance matrix',
           fontsize=8, style='italic', horizontalalignment='center',
           bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor=edge_color, linewidth=1.5),
           zorder=3)
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png', facecolor='white')
    print(f"✓ Figure saved: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', format='pdf', facecolor='white')
    print(f"✓ PDF version saved: {pdf_path}")
    
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate Kalman filter prediction model diagram for thesis'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='PaperLatex/figures/kalman_filter_fig.png',
        help='Output figure path'
    )
    
    args = parser.parse_args()
    
    print(f"Generating Kalman filter prediction model diagram...")
    print(f"  Output: {args.output}")
    
    create_kalman_diagram(args.output)
    print("\n✓ Diagram generation complete!")


if __name__ == '__main__':
    main()

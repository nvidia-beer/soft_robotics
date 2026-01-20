#!/usr/bin/env python3
"""
Combined Figure: All PID Analysis in One Image

Combines:
- Row 1: Classic PID (P, PI, PD, PID)
- Row 2: Neuromorphic PID (P, PI, PD, PID)  
- Row 3: Neuron Count Comparison (250, 500, 1000, 2000)
- Row 4: PES Learning Rate Comparison (0, 1e-5, 1e-4, 1e-3)

Output:
    figures/figure_combined.png
    figures/figure_combined.pdf

Usage:
    python figure_combined.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')


def create_combined_figure():
    """Create combined 4-row figure."""
    
    # Load all data
    classic_file = os.path.join(FIGURES_DIR, 'figure_classic.npz')
    snn_file = os.path.join(FIGURES_DIR, 'figure_snn.npz')
    neurons_file = os.path.join(FIGURES_DIR, 'figure_neuron_count.npz')
    pes_file = os.path.join(FIGURES_DIR, 'figure_pes.npz')
    
    # Check files exist
    for f in [classic_file, snn_file, neurons_file, pes_file]:
        if not os.path.exists(f):
            print(f"Missing: {f}")
            print("Run all figures first: ./create_figure.sh all")
            return
    
    classic = np.load(classic_file, allow_pickle=True)
    snn = np.load(snn_file, allow_pickle=True)
    neurons = np.load(neurons_file, allow_pickle=True)
    pes = np.load(pes_file, allow_pickle=True)
    
    print("Creating combined figure...")
    print(f"  Classic PID: {classic_file}")
    print(f"  SNN PID: {snn_file}")
    print(f"  Neurons: {neurons_file}")
    print(f"  PES: {pes_file}")
    
    # Figure style
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 7,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
    })
    
    # Create 4 rows x 4 columns
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharey='row')
    fig.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.04, 
                        wspace=0.08, hspace=0.20)
    
    # Colors
    colors = {
        'pressure': '#FFAA00',
        'error': '#CC0000',
        'volume': 'black',
        'target': '#CC0000',
    }
    
    # =========================================================================
    # ROW 1: Classic PID
    # =========================================================================
    classic_controllers = ['P', 'PI', 'PD', 'PID']
    
    for col, ctrl in enumerate(classic_controllers):
        ax = axes[0, col]
        
        times = classic[f'{ctrl}_times']
        volumes = classic[f'{ctrl}_volumes']
        errors = classic[f'{ctrl}_errors']
        pressures = classic[f'{ctrl}_pressures']
        
        ax.plot(times, pressures, color=colors['pressure'], linewidth=1.5, 
                label='Pressure', zorder=1)
        ax.plot(times, errors, color=colors['error'], linewidth=1.5, 
                label='Error', zorder=2)
        ax.plot(times, volumes, color=colors['volume'], linewidth=2, 
                label=ctrl, zorder=3)
        ax.axhline(y=2.0, color=colors['target'], linestyle='--', linewidth=1.5, 
                   label='Target', zorder=4)
        
        final_error = abs(2.0 - volumes[-1])
        ax.set_title(f'{ctrl} Control (err: {final_error:.3f})')
        ax.set_ylim(0, 3)
        ax.set_ylabel('Value' if col == 0 else '')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(loc='lower right', fontsize=6)
    
    # Row label
    axes[0, 0].annotate('Classic PID', xy=(-0.25, 0.5), xycoords='axes fraction',
                        fontsize=11, fontweight='bold', rotation=90, va='center')
    
    # =========================================================================
    # ROW 2: SNN PID
    # =========================================================================
    snn_controllers = ['P', 'PI', 'PD', 'PID']
    
    for col, ctrl in enumerate(snn_controllers):
        ax = axes[1, col]
        
        times = snn[f'{ctrl}_times']
        volumes = snn[f'{ctrl}_volumes']
        errors = snn[f'{ctrl}_errors']
        pressures = snn[f'{ctrl}_pressures']
        
        ax.plot(times, pressures, color=colors['pressure'], linewidth=1.5, 
                label='Pressure', zorder=1)
        ax.plot(times, errors, color=colors['error'], linewidth=1.5, 
                label='Error', zorder=2)
        ax.plot(times, volumes, color=colors['volume'], linewidth=2, 
                label=ctrl, zorder=3)
        ax.axhline(y=2.0, color=colors['target'], linestyle='--', linewidth=1.5, 
                   label='Target', zorder=4)
        
        final_error = abs(2.0 - volumes[-1])
        ax.set_title(f'{ctrl} Control (err: {final_error:.3f})')
        ax.set_ylim(0, 3)
        ax.set_ylabel('Value' if col == 0 else '')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(loc='lower right', fontsize=6)
    
    # Row label
    axes[1, 0].annotate('SNN PID', xy=(-0.25, 0.5), xycoords='axes fraction',
                        fontsize=11, fontweight='bold', rotation=90, va='center')
    
    # =========================================================================
    # ROW 3: Neuron Count
    # =========================================================================
    neuron_counts = list(neurons['neuron_counts'])
    
    for col, n_neurons in enumerate(neuron_counts):
        ax = axes[2, col]
        
        times = neurons[f'n{n_neurons}_times']
        volumes = neurons[f'n{n_neurons}_volumes']
        errors = neurons[f'n{n_neurons}_errors']
        pressures = neurons[f'n{n_neurons}_pressures']
        
        ax.plot(times, pressures, color=colors['pressure'], linewidth=1.5, 
                label='Pressure', zorder=1)
        ax.plot(times, errors, color=colors['error'], linewidth=1.5, 
                label='Error', zorder=2)
        ax.plot(times, volumes, color=colors['volume'], linewidth=2, 
                label='PID', zorder=3)
        ax.axhline(y=2.0, color=colors['target'], linestyle='--', linewidth=1.5, 
                   label='Target', zorder=4)
        
        final_error = abs(2.0 - volumes[-1])
        ax.set_title(f'{n_neurons} neurons (err: {final_error:.3f})')
        ax.set_ylim(0, 3)
        ax.set_ylabel('Value' if col == 0 else '')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(loc='lower right', fontsize=6)
    
    # Row label
    axes[2, 0].annotate('Neuron Count', xy=(-0.25, 0.5), xycoords='axes fraction',
                        fontsize=11, fontweight='bold', rotation=90, va='center')
    
    # =========================================================================
    # ROW 4: PES Learning Rate Comparison
    # =========================================================================
    pes_controllers = list(pes['controller_names'])
    
    for col, ctrl in enumerate(pes_controllers):
        ax = axes[3, col]
        
        times = pes[f'{ctrl}_times']
        volumes = pes[f'{ctrl}_volumes']
        errors = pes[f'{ctrl}_errors']
        pressures = pes[f'{ctrl}_pressures']
        
        # Get label from controller name
        if ctrl == 'PES_0':
            label = 'No PES'
        else:
            label = ctrl.replace('PES_', 'lr=')
        
        ax.plot(times, pressures, color=colors['pressure'], linewidth=1.5, 
                label='Pressure', zorder=1)
        ax.plot(times, errors, color=colors['error'], linewidth=1.5, 
                label='Error', zorder=2)
        ax.plot(times, volumes, color=colors['volume'], linewidth=2, 
                label=label, zorder=3)
        ax.axhline(y=2.0, color=colors['target'], linestyle='--', linewidth=1.5, 
                   label='Target', zorder=4)
        
        final_error = abs(2.0 - volumes[-1])
        ax.set_title(f'{label} (err: {final_error:.3f})')
        ax.set_xlabel('Time (s)')
        ax.set_ylim(0, 3)
        ax.set_ylabel('Value' if col == 0 else '')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(loc='lower right', fontsize=6)
    
    # Row label
    axes[3, 0].annotate('PES Learning', xy=(-0.25, 0.5), xycoords='axes fraction',
                        fontsize=11, fontweight='bold', rotation=90, va='center')
    
    # =========================================================================
    # Global title
    # =========================================================================
    fig.suptitle('PID Control Comparison: Classic vs SNN vs Neuron Count vs PES Learning',
                 fontsize=14, fontweight='bold', y=0.97)
    
    # Save
    png_file = os.path.join(FIGURES_DIR, 'figure_combined.png')
    pdf_file = os.path.join(FIGURES_DIR, 'figure_combined.pdf')
    
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nCombined figure saved:")
    print(f"  PNG: {png_file}")
    print(f"  PDF: {pdf_file}")


if __name__ == '__main__':
    create_combined_figure()

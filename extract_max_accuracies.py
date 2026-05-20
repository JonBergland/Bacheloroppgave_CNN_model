#!/usr/bin/env python3
"""
Extract trailing 5-epoch average accuracy scores from ResNet and ViT training results.
Useful for creating tables in papers.
"""

from resnet_visualization import get_max_accuracy_table as get_resnet_table
from resnet_visualization import get_max_accuracy_csv as get_resnet_csv
from vit_visualization import get_max_accuracy_table as get_vit_table
from vit_visualization import get_max_accuracy_csv as get_vit_csv


def print_all_tables():
    """Print formatted tables for both models."""
    print(get_resnet_table())
    print("\n\n")
    print(get_vit_table())


def print_combined_csv():
    """Print combined CSV for both models."""
    print("Combined Results (CSV format):")
    print("=" * 80)
    print(get_resnet_csv())
    print(get_vit_csv())


if __name__ == "__main__":
    print("=" * 80)
    print("FORMATTED TABLES")
    print("=" * 80)
    print_all_tables()
    
    print("\n\n")
    print("=" * 80)
    print("CSV FORMAT (for spreadsheets/papers)")
    print("=" * 80)
    print_combined_csv()

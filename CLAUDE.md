# Claude AI Assistant Documentation for miTCGR

This document provides essential information for Claude AI to effectively assist with the miTCGR project.

## Project Context

**miTCGR** is a deep learning project for predicting microRNA-mRNA interactions using Frequency Chaos Game Representations (FCGRs) and dual-branch CNNs.

## Key Commands & Operations

### Running the Project
```bash
python main.py
```

### Testing Commands
- Check if project has specific test commands in package.json, requirements.txt, or similar files
- Look for pytest, unittest, or custom test scripts

### Linting & Type Checking
- Search for linting configurations (.pylintrc, pyproject.toml, setup.cfg)
- Look for type checking tools like mypy in requirements or config files

## Critical Files to Always Check

1. **config/settings.py** - All hyperparameters and configuration
2. **main.py** - Entry point and pipeline orchestration
3. **core/model.py** - Main dual-branch CNN architecture
4. **utils/DatasetLoader.py** - Data preparation and FCGR generation
5. **utils/FCGR.py** - Sequence-to-image conversion logic

## Architecture Overview

- **Input**: miRNA and mRNA sequences from CSV files
- **Processing**: Convert sequences to FCGR matrices (2D images)
- **Model**: Dual-branch CNN (separate branches for miRNA/mRNA FCGRs)
- **Output**: Binary classification (interaction vs no interaction)

## Datasets
- `data/miraw.csv` - Main dataset
- `data/deepmirtar.csv` - Smaller test dataset

## Development Workflow

1. Always check `config/settings.py` for current configuration
2. Understand data flow: CSV → FCGR matrices → CNN → classification
3. Model variants exist in core/ directory (main, cross-attention, init versions)
4. Training includes early stopping, learning rate scheduling, gradient clipping
5. Results saved to `output/` directory with metrics and visualizations

## Important Notes

- K-mer size determines FCGR matrix resolution (k=6 → 64x64 matrix)
- Training/validation/test split: 70%/15%/15%
- PyTorch-based implementation
- Comprehensive metrics and visualization in utils/

## Update Protocol

Always update this CLAUDE.md file when making significant changes to maintain accurate project understanding.
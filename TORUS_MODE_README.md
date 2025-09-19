# ChiFlow Torus Mode Configuration Guide

This guide explains how to configure and use the torus mode in ChiFlow inference.

## Overview

ChiFlow now supports configurable torus mode that allows you to choose between:
- **Torus Mode** (`torus_mode: true`): Uses high-dimensional torus flow + NERF for backbone generation
- **Standard Mode** (`torus_mode: false`): Uses standard SE(3) flow matching

## Configuration Parameters

### Torus Mode Settings

Add these parameters to your `inference.yaml` configuration:

```yaml
inference:
  # Enable/disable torus flow mode
  torus_mode: true

  # Number of sampling steps for torus flow (only used when torus_mode: true)
  torus_num_steps: 100
```

### Complete Configuration Examples

#### 1. Torus Mode Enabled
```yaml
# Use: python runner/inference.py --config-name inference_torus_example
inference:
  name: chiflow_torus_example
  torus_mode: true
  torus_num_steps: 100
  # ... other settings
```

#### 2. Standard Mode (Torus Disabled)
```yaml
# Use: python runner/inference.py --config-name inference_standard_example
inference:
  name: chiflow_standard_example
  torus_mode: false
  torus_num_steps: 50  # Ignored when torus_mode is false
  # ... other settings
```

## Usage Examples

### Running with Torus Mode
```bash
# Run inference with torus mode enabled
python runner/inference.py --config-name inference_torus_example
```

### Running with Standard Mode
```bash
# Run inference with standard flow matching
python runner/inference.py --config-name inference_standard_example
```

### Command Line Override
You can also override the configuration from command line:

```bash
# Enable torus mode
python runner/inference.py inference.torus_mode=true inference.torus_num_steps=200

# Disable torus mode
python runner/inference.py inference.torus_mode=false
```

## Key Differences

### Torus Mode (`torus_mode: true`)
- Uses high-dimensional torus flow matching
- Generates backbone coordinates directly using NERF
- Better for capturing torsional angle distributions
- Configurable sampling steps via `torus_num_steps`

### Standard Mode (`torus_mode: false`)
- Uses traditional SE(3) flow matching
- Works with existing flow matcher infrastructure
- More stable for certain protein families
- Uses standard `flow.num_t` parameter

## Model Compatibility

- **ChiFlow Model** (`model_name: chiflow`): Supports both modes
- **Other Models**: Automatically use standard mode regardless of `torus_mode` setting

## Logging

The system will log which mode is being used:
```
INFO: Using torus mode for ChiFlow sampling with 100 steps
INFO: Using standard flow matching mode for ChiFlow
INFO: Using SE(3) flow matching mode
```

## Troubleshooting

1. **Torus mode not working**: Ensure `model_name: chiflow` in your model config
2. **Configuration not applied**: Check that parameters are under `inference:` section
3. **Import errors**: Make sure ChiFlow model is properly installed

## Performance Notes

- Torus mode may be slower but can generate more diverse structures
- Standard mode is faster and more stable for most use cases
- Adjust `torus_num_steps` based on your quality vs speed requirements
# Enhanced Evaluation for Leray-Alpha Cylinder Flow

This document describes the enhanced evaluation functionality implemented in `eval.py` for the Leray-Alpha cylinder flow problem.

## Overview

The enhanced evaluation extends the original functionality to include:

1. **L² Error Computation**: Computes relative L² errors for velocity components
2. **Drag, Lift, and Pressure Drop Analysis**: Evaluates time-dependent coefficients
3. **Maximum Value Analysis**: For non-steady flow, computes maximum values (C_D_max, C_L_max, p_diff_max)
4. **Time-Series Plotting**: Generates comprehensive time-series plots
5. **Error Analysis**: Compares computed values against reference values

## Key Features

### 1. L² Error Computation
- Computes relative L² errors for u and v velocity components
- Uses the final time step for comparison with reference data
- Outputs: `l2_error of u` and `l2_error of v`

### 2. Drag, Lift, and Pressure Drop Analysis
- **C_D (Drag Coefficient)**: Time-dependent drag coefficient
- **C_L (Lift Coefficient)**: Time-dependent lift coefficient  
- **p_diff (Pressure Difference)**: Time-dependent pressure difference across cylinder
- **Maximum Values**: C_D_max, C_L_max, p_diff_max for non-steady flow

### 3. Time-Series Plotting

#### Plot 1: Complete Time Interval [t0, t1]
- Shows C_D, C_L, and p_diff over the entire simulation time
- File: `leray_alpha_time_series.pdf`
- Three subplots showing each parameter separately

#### Plot 2: Near Maximum Values
- Zooms in around the maximum values of each parameter
- File: `leray_alpha_zoom_maxima.pdf`
- Shows detailed behavior near peak values
- Includes vertical lines marking exact maximum locations

#### Plot 3: Combined Parameters
- All three parameters on the same graph
- File: `leray_alpha_combined_parameters.pdf`
- Similar to the reference image showing oscillatory behavior

### 4. Error Analysis
- Compares maximum values against reference values
- Computes relative errors for C_D_max, C_L_max, p_diff_max
- **Note**: Reference values are currently placeholders and should be updated with actual reference data

## Usage

### Running the Enhanced Evaluation

```bash
# From the leray_alpha directory
python main.py --config=configs/default.py --workdir=.
```

### Configuration

The evaluation uses the same configuration as training:
- `config.mode = "eval"` in the config file
- Checkpoint path: `./ckpt/{wandb.name}`
- Time windows: `config.training.num_time_windows`

### Output Files

The evaluation generates several output files in `figures/{wandb.name}/`:

1. **Flow Field Plots**:
   - `leray_alpha_cylinder.pdf`: Predicted u, v, p fields
   - `leray_alpha_cylinder_errors.pdf`: Error fields (reference - prediction)
   - `ns_cylinder_reference.pdf`: Reference fields

2. **Time-Series Plots**:
   - `leray_alpha_time_series.pdf`: Complete time interval plots
   - `leray_alpha_zoom_maxima.pdf`: Zoomed plots near maxima
   - `leray_alpha_combined_parameters.pdf`: Combined parameter plot

## Technical Details

### Time-Dependent Computation
- Uses `model.compute_drag_lift(params, t_coords, U_star, L_star)`
- Computes coefficients for each time step
- Handles multiple time windows if configured

### Non-Dimensionalization
- Supports both dimensional and non-dimensional computations
- Automatically handles coordinate and field scaling
- Uses configurable characteristic scales (U_star, L_star, T_star)

### Plotting Features
- High-resolution plots (300 DPI)
- Proper triangulation with cylinder masking
- Color-coded legends and grid lines
- Professional formatting for publication

## Reference Values

**Current Placeholder Values** (should be updated with actual reference data):
- C_D_ref = 3.2
- C_L_ref = 0.5  
- p_diff_ref = 0.1

## Testing

A test script `test_eval.py` is provided to verify basic functionality:

```bash
python test_eval.py --config=configs/default.py
```

This tests:
- Data loading functionality
- Model initialization
- Method signatures
- Basic plotting capabilities

## Future Improvements

1. **Reference Values**: Update with actual reference data for Leray-Alpha cylinder flow
2. **Statistical Analysis**: Add confidence intervals and statistical measures
3. **Animation**: Create animated plots showing time evolution
4. **Export Data**: Save numerical data to CSV/JSON for further analysis
5. **Validation**: Add validation against analytical solutions where available

## Troubleshooting

### Common Issues

1. **Checkpoint Not Found**: Ensure the checkpoint directory exists and contains valid checkpoints
2. **Memory Issues**: Reduce the number of time points (`t_coords`) if memory is limited
3. **Plotting Errors**: Ensure matplotlib is properly installed and backend is configured

### Debug Mode

Add debug prints by modifying the evaluation script:
```python
print(f"Time points: {len(t_coords)}")
print(f"C_D range: [{C_D.min():.4f}, {C_D.max():.4f}]")
```

## Dependencies

- JAX
- NumPy
- Matplotlib
- SciPy
- Flax
- ML Collections
- Weights & Biases (optional) 
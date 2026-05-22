# File Naming Analysis and Recommendations

## Current State

### Test Files
- **Current patterns**: Mix of `test_*.py` and `*_test.py`
- **Preferred standard**: `test_*.py` (more common in Python ecosystem)
- **Already moved to**: `/tests/{validation,performance,integration}/`

### Script Files
- **Current patterns**: Mix of `snake_case` and inconsistent naming
- **Examples**: 
  - `comprehensive_benchmark.py`
  - `debug_bayesian.py`
  - `surface_generation_demo.py`
  - `simple_analysis.py`

### Python Module Files
- **Current patterns**: Generally use `snake_case` (good)
- **Location**: Properly organized in package structure

## Recommended Standards

### Test Files
- **Pattern**: `test_<feature_name>.py`
- **Location**: `/tests/{validation,performance,integration}/`
- **Examples**: 
  - `test_bayesian_optimization.py`
  - `test_prima_algorithms.py`
  - `test_js_validation.py`

### Script/Demo Files
- **Pattern**: `<purpose>_<subject>.py`
- **Examples**:
  - `benchmark_comprehensive.py`
  - `demo_surface_generation.py`
  - `debug_bayesian_optimization.py`
  - `analysis_simple.py`

### Module Files
- **Pattern**: `snake_case.py` (already consistent)
- **Keep current naming**

## Implementation Status
- ✅ Test files moved to organized structure
- ✅ Test file import paths noted for future updates
- 📝 Script naming documented for future standardization
- 📝 Recommendations provided for consistent naming

## Future Actions
1. Standardize script naming when touching files
2. Update any hardcoded import paths
3. Consider creating a style guide for new files
# tsfresh Feature Extraction Categories

This document provides an overview of the main categories of features that tsfresh extracts from time series data. tsfresh automatically extracts hundreds of features organized into several broad categories.

## Overview

tsfresh extracts features using "feature calculators" - functions that compute specific characteristics of a time series. The library includes three presets:
- **Minimal**: ~10 basic summary statistics
- **Efficient**: ~200-300 features (balanced coverage)
- **Comprehensive**: 500+ features (full coverage)

## Main Feature Categories

### 1. **Basic Statistical Features**
Fundamental descriptive statistics of the time series:
- `mean`, `median`, `variance`, `standard_deviation`
- `skewness`, `kurtosis` (distribution shape)
- `variation_coefficient` (coefficient of variation)
- `root_mean_square` (RMS)
- `sum_values`, `abs_energy` (total energy)
- `length` (number of data points)

### 2. **Change and Derivative Features**
Capture how the series changes over time:
- `mean_abs_change`, `mean_change` (average change between consecutive points)
- `mean_second_derivative_central` (second-order changes)
- `absolute_sum_of_changes` (total variation)
- `variance_larger_than_standard_deviation` (distribution spread indicator)

### 3. **Temporal Pattern Features**
Identify patterns and sequences in the time series:
- `longest_strike_below_mean`, `longest_strike_above_mean` (consecutive periods)
- `count_above_mean`, `count_below_mean` (points relative to mean)
- `number_peaks` (local maxima)
- `number_cwt_peaks` (peaks detected via continuous wavelet transform)

### 4. **Autocorrelation Features**
Measure temporal dependencies and periodicity:
- `autocorrelation` (correlation with lagged versions)
- `partial_autocorrelation` (autocorrelation controlling for intermediate lags)
- `c3` (nonlinear autocorrelation measure)
- `fft_coefficient` (Fourier transform coefficients for frequency analysis)

### 5. **Frequency Domain Features**
Derived from spectral analysis:
- `fft_coefficient` (Fourier coefficients at different frequencies)
- `fft_aggregated` (aggregated FFT statistics)
- `spectral_entropy` (entropy of the power spectrum)
- `wavelet_coefficient` (wavelet transform coefficients)

### 6. **Entropy and Complexity Features**
Measure randomness, predictability, and complexity:
- `approximate_entropy` (regularity/predictability measure)
- `sample_entropy` (similar to approximate entropy, more robust)
- `permutation_entropy` (entropy based on ordinal patterns)
- `cwt_coefficient` (continuous wavelet transform features)

### 7. **Location-Based Features**
Identify specific positions in the time series:
- `first_location_of_maximum`, `first_location_of_minimum`
- `last_location_of_maximum`, `last_location_of_minimum`
- `time_reversal_asymmetry_statistic` (temporal asymmetry)

### 8. **Boolean and Duplicate Features**
Binary characteristics:
- `has_duplicate`, `has_duplicate_max`, `has_duplicate_min`
- `is_symmetric` (symmetry checks)

### 9. **Quantile and Percentile Features**
Distribution characteristics at different quantiles:
- `quantile` (values at specific quantiles, e.g., 0.1, 0.2, ..., 0.9)
- `percentage_of_reoccurring_values`
- `percentage_of_reoccurring_datapoints_to_all_datapoints`

### 10. **Linear and Nonlinear Trends**
Trend detection and modeling:
- `linear_trend` (slope and intercept of linear fit)
- `friedrich_coefficients` (coefficients from AR model)
- `ar_coefficient` (autoregressive model coefficients)

### 11. **Aggregation Features**
Summaries computed over different time windows:
- `agg_autocorrelation` (autocorrelation with aggregation)
- `agg_linear_trend` (linear trends with aggregation)
- Various aggregations: `mean`, `var`, `std`, `min`, `max` over chunks

### 12. **Binned Features**
Features computed on binned/discretized versions:
- `binned_entropy` (entropy of binned data)
- Various statistics computed on binned representations

### 13. **Count and Ratio Features**
Counting and proportional measures:
- `count_above`, `count_below` (points above/below thresholds)
- `ratio_beyond_r_sigma` (points beyond r standard deviations)
- `ratio_value_number_to_time_series_length`

### 14. **Complexity and Nonlinearity Features**
Advanced complexity measures:
- `cid_ce` (complexity-invariant distance)
- `lempel_ziv_complexity` (Lempel-Ziv complexity measure)
- `number_cwt_peaks` (complexity via wavelet peaks)

### 15. **Model-Based Features**
Features derived from fitted models:
- `friedrich_coefficients` (parameters from Friedrich's model)
- `ar_coefficient` (autoregressive coefficients)
- `index_mass_quantile` (mass concentration measures)

## Feature Naming Convention

tsfresh features are typically named as:
```
{calculator_name}__{parameter_name}_{parameter_value}
```

For example:
- `fft_coefficient__attr_"real"__coeff_0` - Real part of 0th FFT coefficient
- `quantile__q_0.1` - 10th percentile value
- `autocorrelation__lag_1` - Autocorrelation at lag 1

## Selecting Features

You can control which features are extracted using:

1. **Presets** (in `extract_features.py`):
   - `minimal`: Basic statistics only
   - `efficient`: Balanced set (~200-300 features)
   - `comprehensive`: Full feature set (500+ features)

2. **Custom configuration**: Provide a JSON file with specific feature calculators and parameters (see `--feature-config` in `extract_features.py`)

## References

For the complete list of all available feature calculators and their parameters, see:
- [tsfresh Documentation](https://tsfresh.readthedocs.io/)
- [tsfresh Feature Calculator Reference](https://tsfresh.readthedocs.io/en/latest/text/feature_calculators.html)

## Notes

- The "efficient" preset (default in this project) provides a good balance between feature coverage and computational cost
- Some features may be highly correlated (e.g., mean and median for symmetric distributions)
- PCA is recommended after feature extraction to reduce dimensionality and remove redundancy
- Feature extraction can be computationally intensive for large datasets; consider using `minimal` mode for initial exploration


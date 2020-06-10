import numpy as np
import xarray as xr

# Functions to calculate theoretical relationships

def Spread_Error_Ratio(actual_predictability
                       ,principal_component
                       ,forecast_total_variance
                       ,analysis_total_variance
                       ,forecast_climatology
                       ,analysis_climatology):
    # Calculate ratio of total variance (rot)
    rot=analysis_total_variance/forecast_total_variance
    # Calculate ratio of predictable components (rpc=)
    rpc=actual_predictability/principal_component
    
    nominator = 1/principal_component**2-1
    denominator = rot/principal_component**2 + 1 - 2*np.sqrt(rot)*rpc+ (analysis_climatology-forecast_climatology)**2/(principal_component**2*forecast_total_variance)
    
    return nominator/denominator
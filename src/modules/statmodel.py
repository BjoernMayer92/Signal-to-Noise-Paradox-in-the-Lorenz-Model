# Load Packages
import numpy as np
import xarray as xr




def Generate_Timeseries_Weigel(Nens,Ntim
                               ,analysis_climatology
                               ,forecast_climatology
                               ,analysis_signal_spread
                               ,forecast_signal_ratio
                               ,analysis_total_spread
                               ,forecast_total_spread
                               ):
    
    # Calculate the spread of the forecast signal
    forecast_signal_spread = forecast_signal_ratio*analysis_signal_spread

    # Calculate the forecast and analysis spread from the total and signal spread
    forecast_noise_spread  = np.sqrt(forecast_total_spread**2-forecast_signal_spread**2)
    analysis_noise_spread  = np.sqrt(analysis_total_spread**2-analysis_signal_spread**2)            
    
    # Ensemble Mean Forecast Spread
    forecast_ensmean_spread=np.sqrt(forecast_signal_ratio**2*analysis_signal_spread**2+1/Nens*forecast_noise_spread**2)
    
    # Calculate the expected actual predictability and principal component 
    actual_predictability  = forecast_signal_ratio*analysis_signal_spread**2/(analysis_total_spread*forecast_ensmean_spread) 
    predictable_component  = forecast_ensmean_spread/forecast_total_spread
    
    # Calculate the expected ratio of predictable components
    ratio_of_predictable_components = actual_predictability/predictable_component     #Expected value for ratio of predictable components
    
    
    
    
    # Generate the timeseries for ensemble prediction and verifying analysis 
    # with Ntim timesteps and Nens ensemble members
    
    analysis = np.ones((Ntim))      * np.nan # Initialize analysis array
    forecast = np.ones((Ntim,Nens)) * np.nan # Initialize forecast array
    
    ## Analysis
    # Generate Analysis Signal Component
    analysis_signal = np.random.normal(0,analysis_signal_spread,Ntim)
    # Generate Analysis Noise Component
    analysis_noise  = np.random.normal(0,analysis_noise_spread,Ntim)
    # Save analysis
    analysis        = analysis_climatology + analysis_signal + analysis_noise
    
    ## Forecast
    #Generate Nens ensemble member for forecast
    for ens in range(Nens):
        # Generate forecast error
        forecast_noise = np.random.normal(0,forecast_noise_spread,Ntim)
        # Save forecast
        forecast[:,ens]=forecast_climatology + forecast_signal_ratio*analysis_signal + forecast_noise
    
    # Initialize Dataset
    data=xr.Dataset(data_vars={
        "analysis": (["time"]      ,analysis),
        "forecast": (["time","ens"],forecast)
    },coords={
        "time"                              : (np.arange(Ntim)),
        "ens"                               : (np.arange(Nens)),
        "analysis_climatology"              : analysis_climatology,
        "forecast_climatology"              : forecast_climatology,
        "analysis_signal_spread"            : analysis_signal_spread,
        "forecast_signal_spread"            : forecast_signal_spread,
        "analysis_noise_spread"             : analysis_noise_spread,
        "forecast_noise_spread"             : forecast_noise_spread,
        "forecast_signal_ratio"             : forecast_signal_ratio,               
        "actual_predictability"             : actual_predictability,      
        "ratio_of_predictable_components"   : ratio_of_predictable_components,
        "predictable_component"             : predictable_component   
    })
    
        
    return data
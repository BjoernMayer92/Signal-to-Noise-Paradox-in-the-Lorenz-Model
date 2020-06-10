# Load Packages

import logging
import xarray as xr
import numpy as np
if __name__ == "__main__":
    import numpy as np
    import xarray as xr
    import logging
    import sys
    
    print(sys.argv[0])
    logging.basicConfig(formate="%(asctime)s%(message)s",filename="example.log",level=logging.DEBUG)
    logging.info("Started")
    #xr.load_dataset("../Data/MR30/Seasonal/FullEnsemble/MR30_MSLP_SM_DJF.nc")
    
    
def Signal_Variance(field,dof=0):
    # Calculates the Signal Variance 
    return field.mean(dim="ens").var(dim="time",ddof=dof).rename("signal_variance")
    
def Noise_Variance(field,dof=0):
    # Calculates the Noise Variance
    return field.var(dim="ens",ddof=dof).mean(dim="time").rename("noise_variance") 
    
def Total_Variance(field,dof=0):
    # Calculates the Total Variance
    return field.var(dim=("ens","time"),ddof=dof).rename("total_variance")
    
    
def Analysis_Of_Variance(field,dof=0): 
    # Decompose the total Variability of field in Signal and Noise Variability
    # Field needs to be an xarray and have at least the dimensions "ens" and "time"
    
    total_variance  = Total_Variance(field,dof=dof)
    signal_variance = Signal_Variance(field,dof=dof)
    noise_variance  = Noise_Variance(field,dof=dof)
    
    return total_variance,signal_variance,noise_variance
    
    
def Normalize(field,ensemble,dof=0):
    # Normalize a field by substracting the mean and dividing by the standard deviation
    
    if (ensemble==True): # Calculate stat values over ens and time
        diff = field-field.mean(dim=("ens","time"))
        std  = field.std(dim=("ens","time"),ddof=dof)
        norm = diff / std
    else: # Calculate stat values only over time
        diff = field-field.mean(dim="time")
        std  = field.std(dim="time",ddof=dof)
        norm = diff/std
              
    return norm
              
              
def Spread_Error_Ratio(analysis,forecast,dof=0):
    # Calculates the Spread Error Ratio by calculating Noise and Mean Squared Error
    noise_variance     = Noise_Variance(forecast,dof=dof)
    mean_squared_error = Mean_Squared_Error(analysis,forecast)
    spread_error_ratio = noise_variance/mean_squared_error
    
    return spread_error_ratio

def Mean_Squared_Error(analysis,forecast):
    # Calculates the Mean Squared Error as the sum of the squared difference between 
    # ensemble mean forecast and analysis
    mean_squared_error=((forecast.mean(dim="ens")-analysis)**2).mean(dim="time")
    return mean_squared_error
              
def Ensemble_Correlation(analysis,forecast,dof=0):
    # Calculates the Actual Predictability
    ensemble_mean = forecast.mean(dim="ens")
    analysis_anom = analysis      - analysis.mean(dim="time")
    ensemble_anom = ensemble_mean - ensemble_mean.mean(dim="time")
    
    covariance    = (analysis_anom*ensemble_anom).mean(dim="time")
    forecast_std  = ensemble_mean.std(dim="time",ddof=dof)
    analysis_std  = analysis.std(dim="time",ddof=dof)
              
    return covariance/(forecast_std*analysis_std)
              
def Predictable_Component(forecast,dof=0):
    # Calculates the predictable Component (Eade et al. 2014) 
    # from the signal and noise variance of the forecast
    total_variance  = Total_Variance(forecast,dof=dof)
    signal_variance = Signal_Variance(forecast,dof=dof)
    
    return np.sqrt(signal_variance/total_variance)
    
def Spread_Error_Ratio_Single_Components(analysis,forecast,dof=0):
    # Calculates the Spread Error Ratio based on the theoretical relationship between variances and climatologies (single components)
    total_variance, signal_variance, noise_variance = Analysis_Of_Variance(forecast,dof=dof)
    variance_analysis     = analysis.var(dim="time")
    climatology_forecast  = forecast.mean(dim=("time","ens"))
    climatology_analysis  = analysis.mean(dim="time")
    actual_predictability = Ensemble_Correlation(analysis,forecast,dof=dof)
    
    ratio_total_variance  = variance_analysis/total_variance
    predictable_component = np.sqrt(signal_variance/total_variance)
    
    nominator   = 1/predictable_component**2-1
    
    denominator = ratio_total_variance/predictable_component**2+1-2*np.sqrt(ratio_total_variance)*actual_predictability/predictable_component+(climatology_analysis-climatology_forecast)**2/signal_variance

    return nominator/denominator

def Calculate_All_Statistics(analysis,forecast,dof=0):
    
    actual_predictability           = Ensemble_Correlation(analysis,forecast,dof=dof).rename("actual_predictability")
    predictable_component           = Predictable_Component(forecast,dof=dof).rename("predictable_component")
    model_predictability            = Model_Predictability(forecast).rename("model_predictability")
    
    ratio_of_predictable_components_cor = (actual_predictability / model_predictability.mean(dim="ens")).rename("ratio_of_predictable_components_cor")
    ratio_of_predictable_components_rpc = (actual_predictability/predictable_component).rename("ratio_of_predictable_components_rpc")
    
    forecast_total_variance, forecast_signal_variance, forecast_noise_variance = Analysis_Of_Variance(forecast,dof=dof)
    
    forecast_total_spread           = np.sqrt(forecast_total_variance).rename("forecast_total_spread")
    forecast_signal_spread          = np.sqrt(forecast_signal_variance).rename("forecast_signal_spread")
    forecast_noise_spread           = np.sqrt(forecast_noise_variance).rename("forecast_noise_spread")
    
    analysis_total_spread           = analysis.std(dim=("time"),ddof=dof).rename("analysis_total_spread")
    analysis_climatology            = analysis.mean(dim="time").rename("calculated_analysis_climatology")
    forecast_climatology            = forecast.mean(dim=("time","ens")).rename("calculated_forecast_climatology")
    
    
    spread_error_ratio              = Spread_Error_Ratio(analysis,forecast,dof=dof).rename("pread_error_ratio")
    theoretical_spread_error_ratio  = Spread_Error_Ratio_Single_Components(analysis,forecast,dof=dof).rename("theoretical_spread_error_ratio")
    
    
    data=xr.merge([actual_predictability
                   ,model_predictability
                   ,predictable_component
                   ,ratio_of_predictable_components_rpc
                   ,ratio_of_predictable_components_cor
                   ,forecast_total_spread
                   ,forecast_signal_spread
                   ,forecast_noise_spread
                   ,analysis_total_spread
                   ,analysis_climatology
                   ,forecast_climatology
                   ,spread_error_ratio
                   ,theoretical_spread_error_ratio])
    return data
    
def Model_Predictability(hindcast,leave_out=True):
    # Calculates the model predictability either with excluding the ensemble member (leave_out==True)
    # or with including the ensemble member (leave_out==False)
    
    Nens=hindcast.sizes["ens"]
    
    # Array for saving correlations
    corr=[]
    
    # Loop over all ensemble members
    for ens in range(Nens):
        # Select ensemble member as substitute for the analysis
        substitute_analysis = hindcast.sel(ens=ens)
        if(leave_out==True):
            # Drop the ensemble member which has been chosen to be the substitute for the analysis
            substitute_hindcast = hindcast.drop(dim="ens",labels=ens)
        if(leave_out==False):
            # Take the full ensemble as a hindcast
            substitute_hindcast = hindcast
        # Calculate Ensemble Correlation
        corr.append(Ensemble_Correlation(substitute_analysis,substitute_hindcast))
    
    return xr.concat(corr,dim="ens")


def Normalize_Analyse(data,analysis="analysis",dof=0):
    # Normalize fields and calculates the statistics
    
    
    analysis  = data[analysis]
    hindcast  = data["hindcast"]
    
    analysis_norm  = Normalize(analysis ,ensemble=False,dof=dof)
    hindcast_norm  = Normalize(hindcast ,ensemble=True ,dof=dof)
    
    stat = Calculate_All_Statistics(analysis_norm,hindcast_norm,dof=dof).compute()


    return stat
    
def create_random_permutations(Nens=30,N=100000):
    # Creates an Array of random permutations of the numbers [1 , ... , Nens] with N entries
    
    permutations=np.ones((N,Nens),dtype=np.int8)
    
    for i in range(N):
        permutations[i,:]=np.random.permutation(Nens)
    
    permutations_da = xr.DataArray(permutations,dims=["sample","ens"],coords=[range(N),range(Nens)])
    return permutations_da
    
def Normalize_Scaife_Connected(data,permutations_da,analysis="analysis",Nsample=100,dof=0):
    # Normalize fields and calculates the statistics
    
    permutations = permutations_da.values
    
    analysis  = data[analysis]
    hindcast  = data["hindcast"]
    
    analysis_norm  = Normalize(analysis ,ensemble=False,dof=dof)
    hindcast_norm  = Normalize(hindcast ,ensemble=True ,dof=dof)
    
    data_new = xr.merge([analysis_norm,hindcast_norm])
    
    stat = Calculate_Scaife_Connected(data=data_new,permutations=permutations,Nsample=Nsample).compute()

    return stat
    
    
def Calculate_Scaife_Connected(data, permutations,Nsample=100):
    analysis = data["analysis"]
    hindcast = data["hindcast"]
    
    Nens=data.sizes["ens"]
    
    act = []
    mod = []
    Nlog_sample=int(np.log10(Nsample))
    
    
    for sample in range(Nsample):
        # Take the corresponding permutation array
        permutation_tmp = permutations[sample]
        
        # take the first ensemble member of the permutation to be the substitute for the analysis
        analysis_sub = hindcast.sel(ens=permutation_tmp[0])
        
        act_tmp=[]
        mod_tmp=[]
        
        for mean in range(1,Nens):
            # Create the list of ensemble member indices that should be used
            ensemble_indi = np.arange(1,mean+1)
            # Get the corresponding ensemble members that correspond to permutation with that specific indices
            ensemble_list = permutation_tmp[ensemble_indi]
            
            # Select the corresponding ensembles as a substitute for the hindcast
            hindcast_sub = hindcast.sel(ens=ensemble_list)
            
            # Calculate the actual and Model predictability corresponding to this hindcast
            act_tmp.append(Ensemble_Correlation(analysis    , hindcast_sub,dof=0))
            mod_tmp.append(Ensemble_Correlation(analysis_sub, hindcast_sub,dof=0))
        
        # Combine actual and model predictability for all number of ensemble members
        act.append(xr.concat(act_tmp,dim="number_of_ensemble_members").assign_coords(number_of_ensemble_members=np.arange(1,Nens)))
        mod.append(xr.concat(mod_tmp,dim="number_of_ensemble_members").assign_coords(number_of_ensemble_members=np.arange(1,Nens)))
        
        print("Permutation: "  +str(sample).zfill(Nlog_sample)+"/"+str(Nsample)+" done!",end="\r")
    # return the merged actual and model predictability
    return xr.merge([xr.concat(act,dim="permutation").assign_coords(permutation = np.arange(Nsample)).rename("actual_predictability"),
                     xr.concat(mod,dim="permutation").assign_coords(permutation = np.arange(Nsample)).rename("model_predictability")])
        
    
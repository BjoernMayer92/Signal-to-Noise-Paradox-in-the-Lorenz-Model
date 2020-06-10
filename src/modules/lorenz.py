import numpy as np
import scipy.integrate
import xarray as xr
import pandas as pd
import random
from datetime import datetime
import time
import os



def Lorenz(t,state, rho=28,beta=8.0/3.0,sigma=10.0):
    # Calculates the change of the different components in the lorenz system with 
    # respect to time at point given by state
    
    x, y, z = state # Get the State
    
    x_deriv = sigma * (y - x)
    y_deriv = x * (rho - z)
    z_deriv = x * y - beta * z
    return x_deriv, y_deriv, z_deriv

def Lorenz_Time_Integrate(initial_state, time_interval, out_timestep, rho, beta, sigma):
    # Integrates the Lorenz System forward in time starting from initial state
    # The args-keyword only work with scipy 1.41
    
    # Points in time at which the solution is printed out
    # Small Number is added such that t_eval includes the last step (time_interval[1])
    t_eval=np.arange(time_interval[0],time_interval[1]+0.00000001,out_timestep) 
    
    solution = scipy.integrate.solve_ivp(Lorenz, 
                                         t_span=time_interval, 
                                         t_eval=t_eval,
                                         y0=initial_state,
                                         args=(rho, beta, sigma))
    solution_DA=xr.DataArray(solution.y,
                             coords=[["x","y","z"],solution.t],
                             dims=["dimension","time"])

    
    
    return solution_DA.assign_coords(rho=rho,beta=beta,sigma=sigma)

def Initialize_Hindcast(reference_state, Nens, std_obs=[0.01,0.01,0.01] , std_ens=[0.01,0.01,0.01]):
    # Draw random observation error on all components
    shift_x=np.random.normal(0, std_obs[0])
    shift_y=np.random.normal(0, std_obs[1])
    shift_z=np.random.normal(0, std_obs[2])
    
    observation_state_x = reference_state[0] + shift_x
    observation_state_y = reference_state[1] + shift_y 
    observation_state_z = reference_state[2] + shift_z
    
    initial_conditions=np.ones((Nens,3))
    # Add normal distributed random variables with std corresponding 
    # to ensemble std on top of each component Nens times
    initial_conditions[:,0]=np.random.normal(observation_state_x, std_ens[0], Nens)
    initial_conditions[:,1]=np.random.normal(observation_state_y, std_ens[1], Nens)
    initial_conditions[:,2]=np.random.normal(observation_state_z, std_ens[2], Nens)
    return initial_conditions

def Lorenz_Time_Windows(dataset,dail_stride=10,out_timestep=0.01,days_per_week=7,days_per_mont=30,days_per_seas=90):
    
    last_timestep=dataset.time.isel(time=-1).values
    
    # Calculate all the indices for different leadtimes
    # Select Full days starting with 0.1 in increments of 0.1
    # Added Small number on last_timestep, such that np.arange includes the last timestep
    dail_time_indices = np.arange(out_timestep * dail_stride, last_timestep+10**(-10), out_timestep*dail_stride)
    
    # Calculate running mean where the last entry of the window is the new time coordinate (center=False)
    # Therefore the first entry (initial conditions) is excluded in the calculations
    dm=dataset.rolling(time=dail_stride,center=False).mean().dropna(dim="time").sel(time=dail_time_indices, method="nearest")
    
    wm=dm.rolling(time=days_per_week,center=False).mean().dropna(dim="time")
    mm=dm.rolling(time=days_per_mont,center=False).mean().dropna(dim="time")
    sm=dm.rolling(time=days_per_seas,center=False).mean().dropna(dim="time")
    
    return dm.rename({"time":"leadtime in days"}).rename({"experiment":"time"}),wm.rename({"time":"leadtime in days"}).rename({"experiment":"time"}),mm.rename({"time":"leadtime in days"}).rename({"experiment":"time"}),sm.rename({"time":"leadtime in days"}).rename({"experiment":"time"})


def Create_Hindcast(pool_of_initial_conditions,
                    Nens=30,Nexp=38,
                    length_forecast=20,
                    out_timestep=0.01,
                    rho=28.0,
                    beta=8.0/3.0,
                    sigma=10.0,
                    std_obs=[0.01,0.01,0.01],
                    std_ens=[0.01,0.01,0.01],
                    rho_hin=28.0,
                    beta_hin=8.0/3.0,
                    sigma_hin=10.0):
    # Nens = Number of Ensemble Members
    # Nexp = Number of Experiment (years)
    # Length_Hindcast = Length of Hindcast Experiment in Lorenz-timeunits
    
    Ndim=3 # Number of dimensions
    
    Npool=pool_of_initial_conditions.sizes["time"]
    
    reference_run=[]
    hindcast_run=[]
    
    number_of_zeros=int(np.log10(Nexp))
    # Loop over all Experiments (years)
    for iexp in range(Nexp):
        
        # Draw a randomnumber in the interval [0,Npool-1]
        istart=random.randint(0,Npool-1)
        
        # get the state of the pool of initial conditions as the start date
        reference_state=pool_of_initial_conditions.isel(time=istart)
        
        # Create the initial conditions for the ensemble from the reference state
        
        initial_conditions=Initialize_Hindcast(reference_state.values,
                                               Nens=Nens,
                                               std_obs=std_obs,
                                               std_ens=std_ens)
    
        # Integrate the reference state forward in time
        
        reference_tmp=Lorenz_Time_Integrate(reference_state.values,
                                            time_interval=[0,length_forecast],
                                            out_timestep=out_timestep, 
                                            rho=rho, beta=beta, sigma=sigma)
        
        hindcast_arr=[] # Variable where hindcasts are saved
        
        # Integrate each ensemble member forward in time
        for ens in range(Nens):
            hindcast_tmp = Lorenz_Time_Integrate(initial_conditions[ens],
                                                time_interval=[0,length_forecast],
                                                out_timestep=out_timestep, 
                                                rho=rho_hin, beta=beta_hin, sigma=sigma_hin)
            
            # rename rho, beta and sigma to indicate that they belong to the hindcast not to the reference
            hindcast_arr.append(hindcast_tmp.assign_coords(ens=ens)
                                .rename({"rho":"rho_hin","beta":"beta_hin","sigma":"sigma_hin"}))
        
        # hindcast and reference to list
        hindcast_run.append( xr.concat(hindcast_arr,dim="ens").assign_coords(experiment=iexp) )
        reference_run.append(reference_tmp.assign_coords(experiment=iexp))
        print("Experiment: " +str(iexp).zfill(number_of_zeros) + "/" +str(Nexp) + " done!",end="\r")
    
    # Combine all hindcasts into hindcast_da and all references in reference_da
    reference_da = xr.concat(reference_run , dim="experiment")
    hindcast_da =  xr.concat(hindcast_run, dim="experiment")
    
    # Combine hindcast and reference into one dataset
    result=xr.Dataset({'reference':reference_da,"hindcast":hindcast_da})
    
    return result.assign_coords({"std_obs_x":std_obs[0],
                                 "std_obs_y":std_obs[1],
                                 "std_obs_z":std_obs[2],
                                 "std_ens_x":std_ens[0],
                                 "std_ens_y":std_ens[1],
                                 "std_ens_z":std_ens[2]})
        
def Create_Pool_Of_Initial_Conditions(length_spinup=[0,1000],
                                      length_control=[0,1000],
                                      rho=28.0,
                                      beta=8.0/3.0,
                                      sigma=10.0,
                                      out_timestep=0.01):
    
    spinup=Lorenz_Time_Integrate([1,1,1], length_spinup, out_timestep, rho=rho, beta=beta, sigma=sigma)
    result=Lorenz_Time_Integrate(spinup.isel(time=-1).values,
                                        length_control,out_timestep,rho=rho,beta=beta,sigma=sigma)
    
    return result

def  Create_And_Save_Hindcast(folder,expnumber,pool_of_initial_conditions,
                    Nens=30,Nexp=38,
                    length_forecast=20,
                    out_timestep=0.01,
                    rho=28.0,
                    beta=8.0/3.0,
                    sigma=10.0,
                    std_obs=[0.01,0.01,0.01],
                    std_ens=[0.01,0.01,0.01],
                    rho_hin=28.0,
                    beta_hin=8.0/3.0,
                    sigma_hin=10.0):
    
    start=datetime.now()
    print("Experiment "+str(expnumber)+ " started at " + str(start))
    result=Create_Hindcast(pool_of_initial_conditions=pool_of_initial_conditions,
                    Nens=Nens,Nexp=Nexp,
                    length_forecast=length_forecast,
                    out_timestep=out_timestep,
                    rho=rho,
                    beta=beta,
                    sigma=sigma,
                    std_obs=std_obs,
                    std_ens=std_ens,
                    rho_hin=rho_hin,
                    beta_hin=beta_hin,
                    sigma_hin=sigma_hin)
        
    result.to_netcdf(os.path.join(folder,"Experiment_"+str(expnumber).zfill(3)+".nc"))
    
    end=datetime.now()
    print("Experiment "+str(expnumber)+ " ended at " + str(end) + "! Runtime: "+str((end-start).seconds))
                     
    return start,end

def Reference_To_Analysis(dataset):
    # Adds Gaussian Noise to the different components in the reference run
    # The Result is the Analysis run
    
    reference = dataset["reference"]
    
    length_tim = reference.sizes["time"]
    length_exp = reference.sizes["experiment"]
    
    std_obs_x = reference.std_obs_x
    std_obs_y = reference.std_obs_y
    std_obs_z = reference.std_obs_z
    
    zeros=np.zeros((length_exp,1)) # Define Array of zeros for first elements (the initial conditions)
    
    # Calculate all the errors for the experiments and all timesteps except the first one
    errors_x=np.random.normal(0,std_obs_x,[length_exp,length_tim-1])
    errors_y=np.random.normal(0,std_obs_y,[length_exp,length_tim-1])
    errors_z=np.random.normal(0,std_obs_z,[length_exp,length_tim-1])
    
    # Add zeros to the errors for the first timestep, such that the first timestep
    # has no error (the initialization timestep). For the subsequent time averages
    # the first timestep is omitted.
    
    analysis_error_x = np.concatenate((zeros,errors_x),axis=1)
    analysis_error_y = np.concatenate((zeros,errors_y),axis=1)
    analysis_error_z = np.concatenate((zeros,errors_z),axis=1)
    
    analysis_x = reference.sel(dimension="x")+analysis_error_x
    analysis_y = reference.sel(dimension="y")+analysis_error_y
    analysis_z = reference.sel(dimension="z")+analysis_error_z
    
    analysis=xr.concat([analysis_x,analysis_y,analysis_z],dim="dimension").transpose("experiment","dimension","time").rename("analysis")
    return xr.merge([dataset["reference"],dataset["hindcast"],analysis])
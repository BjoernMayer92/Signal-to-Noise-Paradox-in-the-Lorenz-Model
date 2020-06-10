import time
import multiprocessing as mp
from datetime import datetime
import numpy as np
import sys
import os
import xarray as xr
import argparse

sys.path.append(os.path.join(os.path.abspath(''), '..', 'modules'))
import lorenz

if __name__ == '__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument("--outfolder",        type=str,                    help="Path of folder where output should be saved")
    parser.add_argument("--Nens",             type=int,   default=100,     help="Number of Ensemble Members")
    parser.add_argument("--Nexp",             type=int,   default=100,     help="Number of Experiments") 
    parser.add_argument("--length_forecast",  type=int,   default=20,      help="Integration time in Model Units")
    parser.add_argument("--rho",              type=int,   default=28,      help="Rayleigh Number of Reference Run")
    parser.add_argument("--beta",             type=float, default=8.0/3.0, help="Beta for Reference Run")
    parser.add_argument("--sigma",            type=float, default=10,      help="Pradntl Number for Reference Run")
    parser.add_argument("--rho_hin",          type=int,   default=28,      help="Rayleigh Number for Hindcast")
    parser.add_argument("--beta_hin",         type=float, default=8.0/3.0, help="Beta for Hindcast")
    parser.add_argument("--sigma_hin",        type=float, default=10,      help="Pradntl Number for Hindcast")
    parser.add_argument("--std_obs",          type=float, default=0.01,    help="Standard Deviation of Observation")
    parser.add_argument("--std_ens_min",      type=float, default=0.001,   help="Minimum Standard Deviation of Ensemble")
    parser.add_argument("--std_ens_max",      type=float, default=0.1,     help="Maximum Standard Deviation of Ensemble")
    parser.add_argument("--Nsample",          type=int,   default=501,     help="Number of Samples between min and max of std_ens")
    parser.add_argument("--Ncores",           type=int,   default=20,      help="Number of Cores used for Parallelization")
    parser.add_argument("--out_timestep",     type=int,   default=0.01,    help="Number of Cores used for Parallelization")
    args=parser.parse_args()

    output_folder   = args.__dict__["outfolder"]
    Nens		    = args.__dict__["Nens"]
    Nexp            = args.__dict__["Nexp"]
    length_forecast = args.__dict__["length_forecast"]
    rho             = args.__dict__["rho"]
    beta            = args.__dict__["beta"]
    sigma           = args.__dict__["sigma"]
    rho_hin         = args.__dict__["rho_hin"]
    beta_hin        = args.__dict__["beta_hin"]
    sigma_hin       = args.__dict__["sigma_hin"]
    std_obs         = args.__dict__["std_obs"]
    std_ens_min     = args.__dict__["std_ens_min"]
    std_ens_max     = args.__dict__["std_ens_max"]
    Nsample         = args.__dict__["Nsample"]
    Ncores          = args.__dict__["Ncores"]
    out_timestep    = args.__dict__["out_timestep"]
    
    print("New Set of Sample started with the following parameters: ")
    print("---------------------------------------------------------")
    print("output_folder: "   + str(output_folder))
    print("Nens: "            + str(Nens))
    print("Nexp: "            + str(Nexp))
    print("length_forecast: " + str(length_forecast))
    print("rho: "             + str(rho))
    print("beta: "            + str(beta))
    print("sigma: "           + str(sigma))
    print("rho_hin: "         + str(rho_hin))
    print("beta_hin: "        + str(beta_hin))
    print("sigma_hin: "       + str(sigma_hin))
    print("std_obs: "         + str(std_obs))
    print("std_ens_min: "     + str(std_ens_min))
    print("std_ens_max: "     + str(std_ens_max))
    print("Nsample: "         + str(Nsample))
    print("Ncores: "          + str(Ncores))
    print("out_timestep: "    + str(out_timestep))

    pool_of_initial_conditions=xr.load_dataarray("../../data/InitialConditions/Initial_conditions.nc")
    print("Pool of Initial conditions loaded !")
     
    # If std_ens_min != std_ens_max make a equidistant list in logspace of ens_std 
    if(std_ens_min != std_ens_max):
        ens_std_list = np.logspace(np.log10(std_ens_min),np.log10(std_ens_max),Nsample)
        
    # If std_ens_min == std_ens_max repeat the same experiment with std_ens_min Nsample times
    if(std_ens_min == std_ens_max):
        ens_std_list = np.ones((Nsample))*std_ens_min
    
    print("the different samples have the following standard deviations:")
    print(ens_std_list)
    
    # Prepare Parameters for function:
    parameters=[]
    for sample in range(Nsample):
        std_ens=ens_std_list[sample]
        parameters.append([output_folder,sample,pool_of_initial_conditions,Nens,Nexp,length_forecast,
                           out_timestep,rho,beta,sigma,[std_obs,std_obs,std_obs],[std_ens,std_ens,std_ens],
                           rho_hin,beta_hin,sigma_hin])
    print("Parameters setup done!")
    
    
    pool = mp.Pool(processes=Ncores)
    pool.starmap(lorenz.Create_And_Save_Hindcast, parameters)
    
    pool.close()
    pool.join()
    


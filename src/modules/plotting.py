import numpy as np
import matplotlib.pyplot as plt
import cartopy
import xarray as xr

def getcolorlist_all(levels,cbarname="RdBu",mincolor=0,maxcolor=1):
    base=plt.cm.get_cmap(cbarname)
    color=list(base(np.linspace(mincolor,maxcolor,len(levels)-1)))
    #middle=int(len(color)/2)
    #color[middle]=np.array([1,1,1,1])
    return color

def Calculate_Quantiles(data,dimension,quantiles=[0.00,0.25,0.75,1.00]):
    quantile_array=[]
    for i,quantile in enumerate(quantiles):
        quantile_array.append(data.quantile(q=quantile,dim=dimension))  
    return xr.concat(quantile_array,dim="quantile")

def plot_error_bar(ax,quantile_data,actual=False,quantiles=[0.00,0.25,0.75,1.00]):
    ax.fill_between(quantile_data.number_of_ensemble_members,
                    quantile_data["model_predictability"].sel(quantile=quantiles[0]).squeeze(),
                    quantile_data["model_predictability"].sel(quantile=quantiles[3]).squeeze(),color="blue",alpha=0.1)
    
    ax.fill_between(quantile_data.number_of_ensemble_members,
                    quantile_data["model_predictability"].sel(quantile=quantiles[1]).squeeze(),
                    quantile_data["model_predictability"].sel(quantile=quantiles[2]).squeeze(),color="blue",alpha=0.2)

    if(actual==True):
        ax.fill_between(quantile_data.number_of_ensemble_members,
                        quantile_data["actual_predictability"].sel(quantile=quantiles[0]).squeeze(),
                        quantile_data["actual_predictability"].sel(quantile=quantiles[3]).squeeze(),color="black",alpha=0.1)

        ax.fill_between(quantile_data.number_of_ensemble_members,
                        quantile_data["actual_predictability"].sel(quantile=quantiles[1]).squeeze(),
                        quantile_data["actual_predictability"].sel(quantile=quantiles[2]).squeeze(),color="black",alpha=0.2)
        
def calculate_smallest_distance(data1, data2):
    Nsample = data1.sizes["sample"]
    dim1_norm = ((data1 - data1.mean(dim="sample")) / data1.std(dim="sample"))
    dim2_norm = ((data2 - data2.mean(dim="sample")) / data2.std(dim="sample"))
    
    distance = np.sqrt(dim1_norm**2 +dim2_norm**2).assign_coords(sample = range(Nsample))

    min_distance_index = distance.where(distance == distance.min() ).dropna(dim="sample").sample
    return min_distance_index
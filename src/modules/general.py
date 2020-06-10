import numpy as np


def latex_export_variable(mean, meanstd, filename, significant_digits=2):
    # takes a value and its standard deviation calculates the significant digits and exports the variable into a textfile in latex format such that it can be input in a tex file
    
    total_digits = - (int(np.log10(meanstd)) - significant_digits)
    mean_cr    = str(mean)[0:2+total_digits]
    meanstd_cr = str(meanstd)[0:2+total_digits]
    
    latex_string = "$" + mean_cr +"\ \pm \ " + meanstd_cr+"$"
    
    text_file = open(filename, "w")

    text_file.write(latex_string)

    text_file.close()
    
    print(latex_string)
    
def write_number(number,filename,digits=3):
    
    text_file = open(filename, "w")

    text_file.write(str(number)[0:2+digits])

    text_file.close()
    
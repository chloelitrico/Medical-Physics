import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
import numpy as np

def get_patient_info(filepath):
    #opening file and saving values from text file into a dictionary
    with open(filepath,'r') as f: 
        patient = {}
        for line in f:
            key = line.strip().strip(":")
            value = next(f, None)
            if value is None:
                raise ValueError('Invalid file format, missing key')
            patient[key] = value.strip().split(",")
    return patient

def HU_histogram(patientinfo):
    for key in patientinfo:
        if key == "Hounsfield Units":
            HU = [float(x) for x in patientinfo[key]]
    
    # best fit of data
    (mu, sigma) = norm.fit(HU)

    n, bins, patches = plt.hist(HU, 300, normed=1, facecolor='green', alpha=0.75)

    # add a 'best fit' line
    y = mlab.normpdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)

    #Labels y-axis, x-axis, and adds title
    plt.ylabel("Hounsfield Unit Frequency in GTV-1")
    plt.xlabel("Hounsfield Units")
    plt.title(r'$\mathrm{Hounsfield\ Units\ Found\ Within\ GTV-1: }\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
    plt.grid(True)
    #Prints the graph
    plt.show()
    
    #Statistical values
    
    
    
patientinfo = get_patient_info('Data/HU/001_20080918.txt')
HU_histogram(patientinfo)

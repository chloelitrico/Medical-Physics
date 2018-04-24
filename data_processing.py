%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
#from scipy.stats.distributions import norm
from scipy.stats import norm
import scipy.stats as ss

def get_patient_info(filepath):
    #opening file and saving values from text file into a dictionary
    with open(filepath,'r') as f: 
        patient = {}
        for line in f:
            key = line.strip().strip(":").replace(" ", "")
            value = next(f, None)
            if value is None:
                raise ValueError('Invalid file format, missing key')
            patient[key] = value.strip().split(",")
    return patient


def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(kernel = 'gaussian', bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)


def HU_histogram(patientinfo):

    for key in patientinfo:
        if key == "HounsfieldUnits":
            HU = [float(x) for x in patientinfo[key]]
    HU = np.array(HU)
    
    x_grid = np.linspace(-1100, 500, 500)
    y = [x*100000 for x in kde_sklearn(HU, x_grid, bandwidth=30)]

    fig, ax = plt.subplots()
    ax.plot(x_grid, y, 'r', linewidth=2, alpha=0.75)
    ax.hist(HU, 300, fc='gray', alpha=0.75)
    ax.set_xlim(-1100,500)
    
    fig.set_size_inches(16, 9)

    (mu, sigma) = norm.fit(HU)
    plt.ylabel("Hounsfield Unit Frequency in Rectum", fontsize= 20)
    plt.xlabel("Hounsfield Units", fontsize= 20)
    plt.suptitle("Hounsfield Units Found Within the Rectum:", fontsize = 25)
    plt.title(r'$\mathrm{}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma), fontsize = 20)
    plt.grid(True)
    plt.savefig('graphs_lots/' + patientinfo["SeriesDate"][0])
    
def HU_statiscial_values(patientinfo):
    for key in patientinfo:
        if key == "HounsfieldUnits":
            HU = [float(x) for x in patientinfo[key]]
    HU = np.array(HU)
    
    #calculating mean:
    (mean, sigma) = norm.fit(HU)
    
    #calculating percentage:
    count = 0
    for value in HU:
        if -1100 <= value <= -780:
            count += 1
    percentage = (count/len(HU)) * 100
    
    #determining the mode:
    mode = int(ss.mode(HU)[0])
    
    return mean, mode, percentage
    
patientinfo = get_patient_info('../patient_lotsagas/patientlotsagas_day20_20151027.txt')
HU_histogram(patientinfo)
mean, mode, percentage = HU_statiscial_values(patientinfo)
print("Mean: ", mean, "Mode: " , mode, " Percentage: ", percentage)

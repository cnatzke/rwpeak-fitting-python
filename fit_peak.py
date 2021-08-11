# *************************************************************************************
# * Written by : Connor Natzke
# * Started : July 2021 - Still during the plague..
# * Purpose : Fit peaks and return areas
# * Requirements : Python 3, matplotlib, probably something other stuff numpy,scipy...
# *************************************************************************************
import os
import argparse
import uproot as up
import lmfit as lm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import erfc
from scipy.integrate import quad
from numpy import sqrt, exp, pi


def gaussian(x, centroid, sigma):
    # basic gaussian
    #    return (amplitude / (sqrt(2 * pi) * sigma)) * exp(-(x - centroid)**2 / (2 * sigma**2))
    return exp(-(x - centroid)**2 / (2 * sigma**2))


def skewed_gaussian(x, amplitude, centroid, sigma, R):
    # skewed gaussian
    return amplitude * (1.0 - R / 100.0) * gaussian(x, centroid, sigma)


def peak_bg_function(x, amplitude, centroid, sigma, R, beta):
    # background function
    return R * amplitude / 100.0 * exp((x - centroid) / beta) * erfc((x - centroid) / (sqrt(2.0) * pi) + sigma / (sqrt(2.0) * beta))


def peak_function(x, amplitude, centroid, sigma, R, beta):
    # actual gamma peak
    return skewed_gaussian(x, amplitude, centroid, sigma, R) + peak_bg_function(x, amplitude, centroid, sigma, R, beta)


def step_bg_function(x, amplitude, centroid, sigma, step):
    # basic background function
    return abs(step) * amplitude / 100.0 * erfc((x - centroid) / (sqrt(2.0) * sigma))


def quadratic_bg_function(x, c0, c1, c2, offset):
    return c0 + c1 * (x - offset) + c2 * (x - offset)**2


def total_bg_function(x, amplitude, centroid, sigma, step, c0, c1, c2, offset):
    return step_bg_function(x, amplitude, centroid, sigma, step) + quadratic_bg_function(x, c0, c1, c2, offset)


def total_peak_function(x, amplitude, centroid, sigma, R, beta, step, c0, c1, c2, offset):
    # sum of peak and bg
    return peak_function(x, amplitude, centroid, sigma, R, beta) + total_bg_function(x, amplitude, centroid, sigma, step, c0, c1, c2, offset)


def fit_peak(x, y, y_err, centroid):
    # building composite model
    model = lm.Model(total_peak_function)

    initial_sigma = sqrt(5 + 1.33 * centroid / 1000. + 0.9 * (centroid / 1000)**2) / 2.35
    initial_beta = initial_sigma / 2.
    # set initial parameters
    params = lm.Parameters()
    params.add("amplitude", value=8941.)
    params.add("centroid", value=centroid)
    params.add("sigma", value=initial_sigma, min=0.01, max=10.)
    params.add("beta", value=initial_beta, min=0.000001, max=10., vary=False)
    params.add("R", value=0., min=0.000001, max=100., vary=False)
    params.add("step", value=0.218, min=0., max=100.)
    params.add("c0", value=1.0)
    params.add("c1", value=1.0)
    params.add("c2", value=0.0, vary=False)
    params.add("offset", value=0.0, vary=False)

    fit_results = model.fit(y, x=x, params=params, weights=1.0 / y_err, scale_covar=False)
    return fit_results


def get_peak_area(fit, limit_low, limit_high):
    # integrates peak function and background function. Then finds the difference
    params = fit.params
    peak_area = quad(peak_function, limit_low, limit_high, args=(
        params['amplitude'].value, params['centroid'].value, params['sigma'].value, params['R'].value, params['beta'].value))
    bg_area = quad(total_bg_function, limit_low, limit_high, args=(
        params['amplitude'].value, params['centroid'].value, params['sigma'].value, params['step'].value, params['c0'].value, params['c1'].value, params['c2'].value, params['offset'].value))
    net_area = peak_area[0] - bg_area[0]
    print(peak_area)
    print(bg_area)
    print(f'Peak Area: {peak_area[0]}')
    return


def parse_and_run(args):
    basename = os.path.basename(args.input_file)
    print(f'Found root file: {basename}')
    hist_file = up.open(args.input_file)
    histogram = hist_file['sum_energy']

    # get bins around peak
    peak = 1460
    x = histogram.axis().edges()[peak - 20: peak + 20]
    y = histogram.values()[peak - 20: peak + 20]
    y_err = histogram.errors()[peak - 20: peak + 20]

    fit = fit_peak(x, y, y_err, peak)
    print(fit.fit_report())

    x_eval = np.linspace(peak - 20, peak + 20, 1000)
    y_eval = fit.eval(fit.params, x=x_eval)

    limit_low = peak - 20
    limit_high = peak + 20
    get_peak_area(fit, limit_low, limit_high)

    params = fit.params
    # basic formatting
    sns.set_style('ticks')
    sns.set_context('notebook')
    width, height = plt.figaspect(0.563)  # 16x9
    fig, axes = plt.subplots(num=None, figsize=(width, height), dpi=96)

    axes.step(histogram.axis().edges()[:-1], histogram.values(), where='mid')
    axes.plot(x_eval, y_eval, 'r-', label="fit")

    # individual fit components
    axes.plot(x_eval, total_bg_function(x_eval, params['amplitude'].value, params['centroid'].value, params['sigma'].value, params['step'].value,
              params['c0'].value, params['c1'].value, params['c2'].value, params['offset'].value), 'k--', label="background", alpha=0.7)

    plt.legend()
    plt.tight_layout()
    plt.show()

    return


def main():
    parser = argparse.ArgumentParser(description='GRIFFIN Peak Fitter')

    parser.add_argument('input_file', metavar='file', help='Path to histogram file (.root)')

    args, unknown = parser.parse_known_args()

    parse_and_run(args)
    return


if __name__ == "__main__":
    main()

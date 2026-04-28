import numpy as np
import sys
import os
import re
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", message=".*Attempting to set identical low and high xlims.*")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = REPO_ROOT / "data" / "rf" / "pic" / "ml_data"


GRID_DATASETS = [
	#'rho',
	#'nu_e_el',
	#'nu_e_ex1',
	#'nu_e_ex2',
	'nu_e_ion',
	'nu_i_cx',
	'nu_i_iso',
	#'jtot',
	#'ni',
	#'ne',
	#'E',
	#'flux_i',
	#'flux_e',
	#'phi',
	#'EdotJ',
	#'Ti',
	#'Te',
]

PHASE_DATASETS = [
	#'ivdf_2d',
	#'evdf_2d',
]


#Main loop
def main():

	#File names produced by prep_all_ml_data_new.py
	file_names = []
	for save_name in GRID_DATASETS + PHASE_DATASETS:
		file_names.append(f'{DATA_FOLDER}/ccp_{save_name}.pkl')

	#Plotting logic
	plot_data = False

	#Loading and/or plotting all data
	for fn in file_names:

		print(fn)
		if not os.path.exists(fn):
			raise FileNotFoundError(f'Missing ML dataset: {fn}')
		
		#Loading the data
		with open(fn, 'rb') as f:
			data_all = pickle.load(f)

		validate_dataset(data_all, fn)

		#Directory for plotting
		plot_dir = 'plots_' + fn.split('/')[-1].split('.')[0]
		
		#Checking if the data is 1D or 2D.
		nl = len(data_all[0])

		#Extracting 1d data
		if nl == 4:
			extract_data_1d(data_all, plot=plot_data, plot_dir=plot_dir)
		
		#Extracting 2d data
		elif nl == 5:
			extract_data_2d(data_all, plot=plot_data, plot_dir=plot_dir)
		else:
			raise ValueError(f'Unsupported record length {nl} in {fn}')


def validate_dataset(data_all, file_name):

	if len(data_all) == 0:
		raise ValueError(f'Empty dataset: {file_name}')

	first_len = len(data_all[0])
	if first_len not in (4, 5):
		raise ValueError(f'Unexpected record length {first_len} in {file_name}')

	for idx, data in enumerate(data_all):
		if len(data) != first_len:
			raise ValueError(f'Inconsistent record length in {file_name} at entry {idx}')

		freq = float(data[0])
		press = float(data[1])
		if not np.isfinite(freq) or not np.isfinite(press):
			raise ValueError(f'Non-finite frequency/pressure in {file_name} at entry {idx}')

		xgrid = np.asarray(data[2])
		vals = np.asarray(data[-1])
		if xgrid.ndim != 1:
			raise ValueError(f'xgrid must be 1D in {file_name} at entry {idx}')

		if first_len == 4:
			if vals.ndim != 1:
				raise ValueError(f'1D values must be 1D in {file_name} at entry {idx}')
			if len(xgrid) != len(vals):
				raise ValueError(f'xgrid/value size mismatch in {file_name} at entry {idx}')
		else:
			ygrid = np.asarray(data[3])
			if ygrid.ndim != 1:
				raise ValueError(f'ygrid must be 1D in {file_name} at entry {idx}')
			if vals.ndim != 2:
				raise ValueError(f'2D values must be 2D in {file_name} at entry {idx}')
			if vals.shape != (len(ygrid), len(xgrid)):
				raise ValueError(f'Grid/value shape mismatch in {file_name} at entry {idx}: expected {(len(ygrid), len(xgrid))}, got {vals.shape}')



def extract_data_1d(data_all, plot, plot_dir):

	#Create directory
	if plot and not os.path.exists(plot_dir):
		os.makedirs(plot_dir)	

	#Loading the data
	counter = 1
	for data in data_all:
		print(f'Plot {counter} in {len(data_all)}', end='\r')
		counter += 1

		#Getting the frequency (scalar)
		freq = data[0]

		#Getting the pressure (scalar)
		press = data[1]

		#Getting the grid node locations (vector)
		xgrid = data[2]

		#Getting the density data (vector)
		vals = data[3]

		#Plotting
		if plot:
			pname = plot_dir + '/%.2fMHz_%.2fmTorr.png' % (freq,press)
			plot_line_1d(pname, xgrid, vals, xlabel='x', ylabel='y')




def extract_data_2d(data_all, plot, plot_dir):

	#Create directory
	if plot and not os.path.exists(plot_dir):
		os.makedirs(plot_dir)

	#Loading the data
	counter = 1
	for data in data_all:
		print(f'Plot {counter} in {len(data_all)}', end='\r')
		counter += 1		

		#Getting the frequency (scalar)
		freq = data[0]

		#Getting the pressure (scalar)
		press = data[1]

		#Getting the grid node locations (vector)
		xgrid = data[2]

		#Getting the grid node locations (vector)
		ygrid = data[3]		

		#Getting the density data (vector)
		vals = data[4]

		#Plotting
		if plot:
			pname = plot_dir + '/%.2fMHz_%.2fmTorr.png' % (freq,press)
			plot_surf_2d(pname, xgrid, ygrid, vals, nlev=24, xlabel='x', ylabel='y')





#Plotting a single frame
def plot_line_1d(file_name, X, Y, xlabel = r'x', ylabel = r'y', legend=None, ymin=None, ymax=None):

    #Plot settings
    label_size = 30
    tick_size = 14

    #Shifting axis
    xshift = 0.0
    yshift = 0.0

    #Creating figure
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)

    #Plotting
    ax.plot(X,Y,linewidth=2)

    #Setting limits and labels
    ax.set_xlim([X[0],X[-1]])
    ax.set_ylim([ymin,ymax])
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)

    ax.tick_params(labelsize=tick_size)

    if legend != None:
        ax.legend(legend,fontsize=tick_size)

    #Saving figure
    plt.savefig(file_name)
    plt.close(fig)



#Plotting a single frame
def plot_surf_2d(file_name, X, Y, Z, nlev = 24, xlabel = r'x', ylabel = r'y'):

	#Plot settings
	label_size = 30
	tick_size = 14

	#Shifting axis
	xshift = 0.0
	yshift = 0.0

	#Creating figure
	fig = plt.figure(figsize=(10,8))
	ax = fig.add_subplot(111)

	#Plotting
	if len(Z) != 1:	
		ct = ax.contourf(X,Y,Z,levels=nlev)

		#Setting limits and labels
		ax.set_xlim([X[0],X[-1]])
		ax.set_ylim([Y[0],Y[-1]])
		ax.set_xlabel(xlabel, fontsize=label_size)
		ax.set_ylabel(ylabel, fontsize=label_size)

		ax.tick_params(labelsize=tick_size)

		#Adding the colourbar
		cb = plt.colorbar(ct)
		cb.ax.tick_params(labelsize=tick_size)

	#Saving figure
	plt.savefig(file_name)
	plt.close(fig)




### MAIN FUNCTION ###

###*****************MAIN******************###
if __name__ == "__main__":
    main()

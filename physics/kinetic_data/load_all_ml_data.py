import numpy as np
import sys
import os
import re
import pickle
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message=".*Attempting to set identical low and high xlims.*")


#Main loop
def main():

	#File names
	file_names = []
	file_names.append('ml_data/ccp_E.pkl') #1D grid based data
	file_names.append('ml_data/ccp_Te.pkl') #1D grid based data
	file_names.append('ml_data/ccp_Ti.pkl') #1D grid based data
	file_names.append('ml_data/ccp_ne.pkl') #1D grid based data
	file_names.append('ml_data/ccp_ni.pkl') #1D grid based data
	file_names.append('ml_data/ccp_phi.pkl') #1D grid based data
	file_names.append('ml_data/ccp_ve.pkl') #1D grid based data
	file_names.append('ml_data/ccp_vi.pkl') #1D grid based data
	file_names.append('ml_data/ccp_iedf_surface_1d.pkl') #1D data on an energy grid
	file_names.append('ml_data/ccp_evdf_center_1d.pkl') #1D data on an energy grid
	file_names.append('ml_data/ccp_evdf_2d.pkl') #2D phase space data on x-vx grid
	file_names.append('ml_data/ccp_ivdf_2d.pkl') #2D phase space data on x-vx grid
	file_names.append('ml_data/ccp_iaedf_surface_2d.pkl') #2D data on an energy-angle grid

	#Plotting logic
	plot_data = False

	#Loading and/or plotting all data
	for fn in file_names:

		print(fn)
		
		#Loading the data
		with open(fn, 'rb') as f:
			data_all = pickle.load(f)

		#Directory for plotting
		plot_dir = 'plots_' + fn.split('/')[-1].split('.')[0]
		
		#Checking if the data is 1D or 2D.
		nl = len(data_all[0])

		#Extracting 1d data
		if nl == 4:
			extract_data_1d(data_all, plot=True, plot_dir=plot_dir)
		
		#Extracting 2d data
		elif nl == 5:
			extract_data_2d(data_all, plot=True, plot_dir=plot_dir)



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
		#print(xgrid)
		#Computing grid spacing
		ncells = len(xgrid) - 1

		#Writing to file
		with open('grid_spacing_data.txt', 'a') as f:
			f.write(f'{freq:.2f} MHz, {press:.2f} mTorr, number of cells: {ncells}\n')

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
    #ax.set_ylim([Y[-1],Y[0]])
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
		#ax.set_ylim([Y[0],Y[-1]])
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

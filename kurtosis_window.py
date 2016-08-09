from __future__ import division
import numpy as np
from KabschAlign import *
from joblib import Parallel, delayed
from sys import stdout
import os

def update_progress(progress):
    barLength = 40 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def weightedKurtosis(coords, weights):
    tmp = np.average(coords, axis=0, weights=weights);
    devs = coords - tmp;

    #   Compute and weight
    vardevs = np.multiply((devs**2).transpose(2,1,0), weights).transpose(2,1,0);
    kdevs = np.multiply((devs**4).transpose(2,1,0), weights).transpose(2,1,0);

    scale = (coords.shape[2]*coords.shape[1]);
    sm = np.sum(vardevs);
    wVar = sm/scale;
    wK = scale*np.sum(kdevs)/(sm**2);

    #wK = window's kurt., 
    #wVar = window's var.,
    #caKurt = kurt. per residue
    #caVar = var. per residue
    
    return wK, wVar;

def weightedIterativeMeans(coords, weights, eps, maxIter=4):

	Ns = np.shape(coords)[0];
	Na = np.shape(coords)[2];
		
	avgCoords = [];			# track average coordinates
	kalign = KabschAlign();		# initialize for use

	ok = 0;				# tracking convergence of iterative means
	itr = 1; 			# iteration number

	while not(ok):
		mnC = np.average(coords, weights=weights, axis=0); 
		for i in range(0,Ns):
			fromXYZ = coords[i];
			[R, T, xRMSD, err] = kalign.kabsch(mnC, fromXYZ);
			tmp = np.tile(T.flatten(), (Na,1)).T;
			pxyz = np.dot(R,fromXYZ) + tmp;
			coords[i] = pxyz;
		newMnC = np.average(coords, weights=weights, axis=0); 
		err = math.sqrt(np.linalg.norm(mnC.flatten()-newMnC.flatten()));
		if err <= eps or itr == maxIter:
			ok = 1;
		itr = itr + 1;
	return coords;

def _computekurt(halflife, shape, i):
    numres, numsamples = shape;
    numres = numres / 3;
    kurtWindow  = np.memmap('.memmapped/kW.array', dtype='float64', mode='r+', shape=(numsamples - 1));    #   Kurtosis
    varWindow   = np.memmap('.memmapped/vW.array', dtype='float64', mode='r+', shape=(numsamples - 1));    #   Variance
    init_coord  = np.memmap('.memmapped/coords.array', dtype='float64', mode='r', shape=shape);    #   data

    kalign = KabschAlign();
    #   If i < halflife, creates weights from 0 -> i - 1, else 0 -> halflife - 1
    a = np.min( [halflife, i] );
    ind = np.arange(a-1, -1, -1);    
    tau = (halflife)/np.log(2);    
    alpha = 1 - np.exp(-(tau**-1));	#	From paper
    weights = alpha / np.exp( ind / tau );
    weights /= np.sum(weights);

    #   Creates window boundaries and grabs data
    ubound = i;
    lbound = i - a;
    win_coords = np.array(init_coord[:,lbound:ubound]);
    
    #   Returns the aligned mean structure as shape: [3,numRes]
    win_coords = win_coords.reshape((3,numres,a),order='F').transpose(2,0,1);
    mean_struct = weightedIterativeMeans( win_coords, weights, 1e-4 )[-1];
    #   Aligns each structure in the window on the computed weighted mean
    new_struct = np.empty((a,3,numres));
    for j in range(a):
        R, T, junk, junk2 = kalign.kabsch(mean_struct, win_coords[j]);
        win_coords[j] = R.dot(win_coords[j]) + T.reshape(-1,1);

    kurtWindow[i-1],varWindow[i-1],resKurt[:,i-1],resVar[:,i-1] = weightedKurtosis(win_coords, weights);
    kurtWindow.flush();
    varWindow.flush();
    resKurt.flush();
    resVar.flush();

    update_progress(i/(numsamples - 1));

def windowKurtosis(coords):
    #	Kurtosis Sliding Window Computation

	    #	coords = 3*numRes x numSamples

    shape = coords.shape;
    numres, numsamples = shape;
    assert( numres % 3 == 0 );
    numres = numres / 3;
    print numres;
    halflife = 100;

    #   Create some arrays to throw everything in
    mapcoords  = np.memmap('.memmapped/coords.array', dtype='float64', mode='w+', shape=shape);    #   data
    mapcoords[:,:] = coords;
    mapcoords.flush();
    kurtWindow  = np.memmap('.memmapped/kW.array', dtype='float64', mode='w+', shape=(numsamples - 1));    #   Kurtosis
    varWindow   = np.memmap('.memmapped/vW.array', dtype='float64', mode='w+', shape=(numsamples - 1));    #   Variance

    del mapcoords, kurtWindow, varWindow, resKurt, resVar;

    Parallel(n_jobs=8)(delayed(_computekurt)(halflife, shape, i) for i in range(1,numsamples));

    kurtWindow  = np.memmap('.memmapped/kW.array', dtype='float64', mode='r', shape=(numsamples - 1));    #   Kurtosis
    varWindow   = np.memmap('.memmapped/vW.array', dtype='float64', mode='r', shape=(numsamples - 1));    #   Variance

    coords = coords.reshape((3,numres,-1),order='F').transpose(2,0,1);

    tmp = np.mean(coords, axis=0);
    devs = coords - tmp;

    #   Compute and weight
    vardevs = devs**2;
    kdevs = devs**4;

    #   Weighted average of variance and kurtosis over window per residue
    caVar   = np.sum(vardevs, axis=0);
    caKurt  = np.divide(np.sum(kdevs, axis=0), caVar**2);

    #   Averages per resdiue
    caVar   = np.mean(caVar, axis=0);
    caKurt  = np.mean(caKurt, axis=0);

    np.save('{0}_windowKurtosis.npy'.format('hivp'), kurtWindow);
    np.save('{0}_windowVariance.npy'.format('hivp'), varWindow);
    np.save('{0}_residueKurtosis.npy'.format('hivp'), caKurt);
    np.save('{0}_residueVariance.npy'.format('hivp'), caVar);


if __name__ == '__main__':
    data = np.load('hivp_coords.npy')[:,::10];
    windowKurtosis(data);


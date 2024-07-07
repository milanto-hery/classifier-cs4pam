#!/usr/bin/env python
# coding: utf-8

import pickle
import time
import sys
import os

import numpy as np
import numpy.random as rand
import scipy.fftpack as spfft
import cvxpy as cvx
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit


# Generating dct and idct for 2D because SciPy does not provide the 2D versions

def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


class CS:
    
    def compress(self, spectrogram, R, seed):
        """
        Function for compressing spectrograms by sampling from a specific percentage of the original.
        
        R: The percentage value from which we would like to sample from the original
        """
       
        # Get the height and width of the image spectrogram    
        ny,nx,_ =spectrogram.shape
    
        # Get the number of samples
        m = round(nx * ny * R) # e.g: R=0.25 => 25% of samples
        rand.seed(seed)
        ri_vector = rand.choice(nx * ny, m, replace=False) # random sample of indices
        y = spectrogram.T.flat[ri_vector]
        
        # create dct matrix operator using Kronecker product.
        # this approach may lead to memory errors for large ny*nx
        A = np.kron(
            spfft.idct(np.identity(nx), norm='ortho', axis=0),
            spfft.idct(np.identity(ny), norm='ortho', axis=0)
            )
        A = A[ri_vector,:] # same as phi times psi in the theoretical part A=|phi*\psi  

        return y, A
    
    def reconstruct(self, y, A, ny, nx, solver):
        """
        Reconstructs the signal by using differents solvers to approximate the original signal.
        
        Problem: y=As

        input:
            y (ndarray): Compressed measurements.
            A (ndarray): Sensing matrix.

        Output:
            s: sparse solution -> x: reconstructed signal after inverse trasformation of DCT2
        """
        
        if solver == 'lasso':
            
            prob = Lasso(alpha=1e-5)
            prob.fit(A, y)
            s = prob.coef_
            x_lasso = idct2(s.reshape(nx, ny)).T
            x = np.reshape(x_lasso, (ny, nx,1))
            
            return x
        
        elif solver == 'cvx':
            
            s = cvx.Variable(nx*ny)
            objective = cvx.Minimize(cvx.norm(s, 1))
            constraint = [A*s == y]
            prob = cvx.Problem(objective, constraint)
            res = prob.solve(verbose=False, solver='SCS')
            s1 = np.array(s.value).squeeze()
            x0 = idct2(s1.reshape((nx, ny)).T)
            x = np.reshape(x0, (ny, nx,1))
                      
            return x
        
        elif solver == 'omp':
                                                             
            prob = OrthogonalMatchingPursuit()
            prob.fit(A, y)
            s= prob.coef_
            x_omp = idct2(s.reshape((nx, ny)).T)
            x = np.reshape(x_omp, (ny, nx,1))
            
            return x
        else:
            raise ValueError("Please specify solver!!!.")
            
    def save_data_to_pickle(self, X_original_S, X_compressed_S, X_reconstructed_S, Y_values, start_index, end_index, 
                            saved_output):
        '''
        Save all of the data to pickle files.

        '''
        if not os.path.exists(saved_output):
            os.makedirs(saved_output)                                       
        
        outfile = open(os.path.join(saved_output, 'X_original_S#{}_{}.pkl'.format(start_index, end_index)),'wb')
        pickle.dump(X_original_S, outfile, protocol=4)
        outfile.close()
        
        outfile = open(os.path.join(saved_output, 'X_compressed_S#{}_{}.pkl'.format(start_index, end_index)),'wb')
        pickle.dump(X_compressed_S, outfile, protocol=4)
        outfile.close()

        outfile = open(os.path.join(saved_output, 'X_reconstructed_S#{}_{}.pkl'.format(start_index, end_index)),'wb')
        pickle.dump(X_reconstructed_S, outfile, protocol=4)
        outfile.close()
        
        outfile = open(os.path.join(saved_output, 'Y#{}_{}.pkl'.format(start_index, end_index)),'wb')
        pickle.dump(Y_values, outfile, protocol=4)
        outfile.close()
        
        
        print(f'All data saved succesfully to {saved_output}')
        print('===============================================================')

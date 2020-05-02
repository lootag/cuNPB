#include <iostream>
#include <eigen3/Eigen/Dense>
#include "preprocessing.h"
#include "enumerators.h"
#include <vector>

Eigen::MatrixXf GetGradient(Eigen::MatrixXf Train, Eigen::MatrixXf labels, float sigma, float l)
{
    Eigen::MatrixXf gradient(2,1);
    kernel_type type_standard = standard;
    kernel_type type_dK_dsigma = dK_dsigma;
    kernel_type type_dK_dl = dK_dl;
    Eigen::MatrixXf Kernel = GetKernel(Train, Train, sigma, l, type_standard);   
    Eigen::MatrixXf dK_dsigma = GetKernel(Train, Train, sigma, l, type_dK_dsigma);   
    Eigen::MatrixXf dK_dl = GetKernel(Train, Train, sigma, l, type_dK_dl);   
    Eigen::MatrixXf K_inv = Kernel.inverse();
    
    Eigen::MatrixXf dsigma = 0.5*Multiply(labels.transpose(), K_inv.transpose());
    dsigma = Multiply(dsigma, dK_dsigma.transpose());
    dsigma = Multiply(dsigma, K_inv.transpose());
    dsigma = Multiply(dsigma, labels.transpose());
    float dsigmaf = dsigma(0,0);
    float trace_dsigma = 0.5*Multiply(K_inv, dK_dsigma.transpose()).trace();
    dsigmaf -= trace_dsigma;
    
    Eigen::MatrixXf dl = 0.5*Multiply(labels.transpose(), K_inv.transpose());
    dl = Multiply(dl, dK_dl.transpose());
    dl = Multiply(dl, K_inv.transpose());
    dl = Multiply(dl, labels.transpose());
    float dlf = dl(0,0);
    float trace_dl = 0.5*Multiply(K_inv, dK_dl.transpose()).trace();
    trace_dl = 100;
    dlf -= trace_dl;
    gradient(0,0) = dsigmaf; 
    gradient(1,0) = dlf;
    
    return gradient;
}

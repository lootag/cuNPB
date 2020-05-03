#include <iostream>
#include <eigen3/Eigen/Dense>
#include <math.h>
#include "model.h"
#include "preprocessing.h"
#include "enumerators.h"

void model::PredictVariance()
{
    Eigen::MatrixXf X_Train = model::get_X_Train();
    Eigen::MatrixXf Y_Train = model::get_Y_Train();
    Eigen::MatrixXf X_Test = model::get_Y_Test();
    float sigma = model::get_sigma();
    float l = model::get_l();
    
    kernel_type type_standard = standard;
    Eigen::MatrixXf K_Train_Train = GetKernel(X_Train, X_Train, sigma, l, type_standard);
    Eigen::MatrixXf K_Test_Test = GetKernel(X_Test, X_Test, sigma, l, type_standard);
    Eigen::MatrixXf K_Train_Test = GetKernel(X_Train, X_Test, sigma, l, type_standard);
    Eigen::MatrixXf K_Test_Train = GetKernel(X_Test, X_Train, sigma, l, type_standard);
    Eigen::MatrixXf I;
    I.setIdentity(K_Train_Train.rows(), K_Train_Train.cols());
    
    Eigen::MatrixXf MiddleTerm = K_Train_Train + (pow(sigma, 2) * I);
    Eigen::MatrixXf MiddleTerm_inv = MiddleTerm.inverse();
    Eigen::MatrixXf RH = Multiply(K_Test_Train, MiddleTerm_inv.transpose());
    RH = Multiply(RH, K_Train_Test);
    Eigen::MatrixXf Variance = K_Test_Test - RH + (pow(sigma, 2) * I);
    model::set_Sigma_2(Variance);
}
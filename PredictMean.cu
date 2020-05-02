#include <iostream>
#include <math.h>
#include "model.h"
#include "preprocessing.h"
#include "enumerators.h"
#include <eigen3/Eigen/dense>

void model::PredictMean()
{
    kernel_type type_standard = standard;
    Eigen::MatrixXf X_Train = model::get_X_Train();
    Eigen::MatrixXf Y_Train = model::get_Y_Train();
    Eigen::MatrixXf X_Test = model::get_X_Test();
    float sigma = model::get_sigma();
    float l = model::get_l();
    Eigen::MatrixXf K_Test_Train = GetKernel(X_Test, X_Train, sigma, l, type_standard);
    Eigen::MatrixXf K_Train_Train = GetKernel(X_Train, X_Train, sigma, l, type_standard);
    Eigen::MatrixXf MiddleTerm = K_Train_Train + (pow(sigma, 2) * Eigen::MatrixXf::Identity(K_Train_Train.rows(), K_Train_Train.col())); 
    Eigen::MatrixXf MiddleTerm_inv = MiddleTerm.inverse();
    Eigen::MatrixXf mu = Multiply(K_Test_Train, MiddleTerm_inv.transpose());
    mu = Multiply(mu, X_Train.transpose());
    model::set_mu(mu);

}
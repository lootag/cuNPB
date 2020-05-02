#include <iostream>
#include <string>
#include <math.h>
#include "model.h"
#include "optimizers.h"
#include <eigen3/Eigen/Dense>

void model::Train()
{
    
    float alpha = model::get_alpha();
    float beta1 = model::get_beta1();
    float beta2 = model::get_beta2();
    float tolerance = 0.008;
    float max_iterations = 20000;
    Eigen::MatrixXf X_Train = model::get_X_Train();
    Eigen::MatrixXf Y_Train = model::get_Y_Train();
    std::vector<float> parameters = Adam(alpha, beta1, beta2, X_Train, Y_Train, tolerance, max_iterations);
    model::set_sigma(parameters[0]);
    model::set_l(parameters[1]);
    
    
}
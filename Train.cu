#include <iostream>
#include <string>
#include "model.h"
#include "optimizers.h"
#include <eigen3/Eigen/Dense>

std::vector<float> model::Train()
{
    std::vector<float> parameters{0, 0};
    model::set_sigma_2(3);
    model::set_l_2(3);

    parameters[0] = model::get_sigma_2();
    parameters[1] = model::get_l_2();
    return parameters;
    
}
#include <iostream>
#include <string>
#include "model.h"
#include "optimizers.h"
#include <eigen3/Eigen/Dense>

std::vector<float> model::Train()
{
    model::set_sigma_2(3);
    model::set_l_2(3);

    std::cout << std::to_string(model::get_sigma_2() + model::get_l_2()) << std::endl;
}
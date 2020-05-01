#ifndef __OPTIMIZERS_H__
#define __OPTIMIZERS_H__
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>

std::vector<float> Adam(float alpha, float beta1, float beta2, Eigen::MatrixXf Train);
#endif
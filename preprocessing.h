#ifndef __PREPROCESSING_H__
#define __PREPROCESSING_H__
#include <eigen3/Eigen/Dense>
#include <vector>

Eigen::MatrixXf from_csv(std::string path, int rows, int cols);
Eigen::MatrixXf Multiply(Eigen::MatrixXf Left, Eigen::MatrixXf Right);

#endif
#ifndef __PREPROCESSING_H__
#define __PREPROCESSING_H__
#include <eigen3/Eigen/Dense>
#include <vector>
#include "enumerators.h"

Eigen::MatrixXf from_csv(std::string path, int rows, int cols);
void to_csv(Eigen::MatrixXf Matrix, std::string file);
Eigen::MatrixXf Multiply(Eigen::MatrixXf Left, Eigen::MatrixXf Right);
Eigen::MatrixXf GetKernel(Eigen::MatrixXf A, Eigen::MatrixXf B, float sigma, float l, kernel_type type);
Eigen::MatrixXf GetGradient(Eigen::MatrixXf Train, Eigen::MatrixXf labels, float sigma, float l);
#endif
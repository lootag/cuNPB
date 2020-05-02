#include <iostream>
#include <random>
#include <math.h>
#include <string>
#include <eigen3/Eigen/Dense>
#include "preprocessing.h"
#include "optimizers.h"

std::vector<float> Adam(float alpha, float beta1, float beta2, Eigen::MatrixXf Train, Eigen::MatrixXf labels, float tolerance, int maximum_iterations)
{
    std::vector<float> params {0, 0};
    int n_param = 2;
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(1,1);
    Eigen::MatrixXf m(n_param, 1);
    Eigen::MatrixXf v(n_param, 1);
    for(int row = 0; row != m.rows(); row++)
    {
        for(int col = 0; col != m.cols(); col++)
        {
            m(row, col) = 0;
        }
    }
    for(int row = 0; row != v.rows(); row++)
    {
        for(int col = 0; col != v.cols(); col++)
        {
            v(row, col) = 0;
        }
    }
    Eigen::MatrixXf m_hat = m;
    Eigen::MatrixXf v_hat = v;
    float l = distribution(generator);
    float sigma = distribution(generator);
    int iteration = 0;
    float epsilon = pow(10, -8);
    Eigen::MatrixXf gradient = GetGradient(Train, labels, sigma, l);
    Eigen::MatrixXf gradient_2(2,1);
    float percentage_change = 1000;
    while((!(percentage_change <= tolerance) && !(iteration >= maximum_iterations)) || iteration <= 1000)
    {
        iteration += 1;
        std::cout << "iteration number " + std::to_string(iteration) << std::endl;
        gradient = GetGradient(Train, labels, sigma, l);
        for(int row = 0; row != gradient_2.rows(); row++)
        {
            for(int col = 0; col != gradient_2.cols(); col++)
            {
                gradient_2(row, col) = gradient(row, col)*gradient(row, col);
            }
        }
        m = beta1 * m + (1 - beta1) * gradient;
        v = beta2 * v + (1 - beta2) * gradient_2;
        m_hat = m/(1 - pow(beta1, iteration));
        v_hat = v/(1 - pow(beta2, iteration));
        float sum_this = l + sigma;
        sigma = sigma  - alpha * m_hat(0,0) / (sqrt(v_hat(0,0) + epsilon));
        l = l  - alpha * m_hat(1,0) / (sqrt(v_hat(1,0) + epsilon));
        float sum_next = l + sigma;
        percentage_change = std::abs(sum_next - sum_this)/sum_this * 100;
        std::cout << "Variation of " + std::to_string(percentage_change) + " %" << std::endl; 

    }
    params[0] = sigma;
    params[1] = l;
    
    return params;
    
}

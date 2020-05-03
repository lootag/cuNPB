#include <iostream>
#include "preprocessing.h"
#include "enumerators.h"
#include "optimizers.h"
#include "model.h"
#include <string>
#include <unistd.h>
#include <stdlib.h>
#include <eigen3/Eigen/Dense>

int main(int argc, char *argv[])
{
    if(argc < 8)
    {
        std::cout << "You're missing some arguments. Please read the docs, then try again." << std::endl;
        exit(1);
    }
    
    std::string train = argv[1];
    std::string test = argv[4];
    std::string labels = argv[6];
    int train_rows = atoi(argv[2]);
    int train_cols = atoi(argv[3]);
    int test_rows = atoi(argv[5]);
    int labels_cols = atoi(argv[7]);
    char *directory = "/content/data"; 
    int ret;
    ret = chdir(directory);
    Eigen::MatrixXf X_Train = from_csv(train, train_rows, train_cols);
    Eigen::MatrixXf X_Test = from_csv(test, test_rows, train_cols);
    Eigen::MatrixXf Y_Train = from_csv(labels, train_rows, labels_cols);
    std::string mean = "mean.csv";
    std::string variance = "variance.csv";

    
    /*
    Eigen::MatrixXf Result = Multiply(Left, Right.transpose());
    for(int row = 0; row != Result.rows(); row++)
    {
        for(int col = 0; col != Result.cols(); col++)
        {
            std::cout << Result(row, col) << std::endl;
        }
    }
    */
    Eigen::MatrixXf A(3,4);
    Eigen::MatrixXf B(3,1);
    float tmp = 1;
    for(int row = 0; row != A.rows(); row++)
    {
        for(int col = 0; col != A.cols(); col++)
        {
            A(row, col) = tmp;
            tmp += 1;
        }
    }
    tmp = 1;
    for(int row = 0; row != B.rows(); row++)
    {
        for(int col = 0; col != B.cols(); col++)
        {
            B(row, col) = tmp;
            tmp += 1;
        }
    }
    /*
    kernel_type type_standard = standard;
    Eigen::MatrixXf Result = GetKernel(A, A, 1, 1, type_standard);
    std::cout << "Standard Case" << std::endl;
    for(int row = 0; row != Result.rows(); row++)
    {
        for(int col = 0; col != Result.cols(); col++)
        {
            std::cout << Result(row, col) << std::endl;
        }
    }
    
    kernel_type type_dK_dsigma = dK_dsigma;

    Eigen::MatrixXf dK_dsigma = GetKernel(A, A, 1, 1, type_dK_dsigma);
    std::cout << "Derivative of K with respect to sigma" << std::endl;
    for(int row = 0; row != dK_dsigma.rows(); row++)
    {
        for(int col = 0; col != dK_dsigma.cols(); col++)
        {
            std::cout << dK_dsigma(row, col) << std::endl;
        }
    }

    kernel_type type_dK_dl2 = dK_dl;

    Eigen::MatrixXf dK_dl2 = GetKernel(A, A, 1, 1, type_dK_dl2);
    std::cout << "Derivative with respect to l2" << std::endl;
    for(int row = 0; row != dK_dl2.rows(); row++)
    {   
        for(int col = 0; col != dK_dl2.cols(); col++)
        {
            std::cout << dK_dl2(row, col) << std::endl;
        }
    }
    */
    /*
    Eigen::MatrixXf Gradient = GetGradient(A, B , 1, 1);    
    for(int row = 0; row != Gradient.rows(); row++)
    {
        for(int col = 0; col != Gradient.cols(); col++)
        {
            std::cout << Gradient(row, col) << std::endl;
        }
    }
    */

    
    float alpha = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.999;
    /*
    std::vector<float> parameters = Adam(alpha, beta1, beta2, A, B, 0.008, 10000);
    for(int index = 0; index != parameters.size(); index++)
    {
        std::cout << parameters[index] << std::endl; 
    }
    */
    model Model(X_Train, Y_Train, X_Test, alpha, beta1, beta2);
    Model.Train();
    Model.PredictMean();
    to_csv(Model.get_mu(), mean); 
    std::cout << "The mean is" << std::endl;
    for(int row = 0; row != Model.get_mu().rows(); row ++)
    {
        for(int col = 0; col != Model.get_mu().cols(); col++)
        {
            std::cout << std::to_string(Model.get_mu()(row, col)) << std::endl; 
        }
    }

    Model.PredictVariance();
    to_csv(Model.get_Sigma_2(), variance);
    std::cout << "The variance is" << std::endl;
    for(int row = 0; row != Model.get_Sigma_2().rows(); row++)
    {
        for(int col = 0; col != Model.get_Sigma_2().cols(); col ++)
        {
            std::cout << std::to_string(Model.get_Sigma_2()(row,col)) << std::endl;
        }
    }

    std::cout << "X_Train" << std::endl;
    for(int row = 0; row != X_Train.rows(); row++)
    {
        for(int col = 0; col != X_Test.cols(); col++)
        {
            std::cout << std::to_string(X_Train(row,col)) << std::endl;
        }
    }

    std::cout << "X_Test" << std::endl;
    for(int row = 0; row != X_Test.rows(); row++)
    {
        for(int col = 0; col != X_Test.cols(); col++)
        {
            std::cout << std::to_string(X_Test(row,col)) << std::endl;
        }
    }

    std::cout << "Y_Train" << std::endl;
    for(int row = 0; row != Y_Train.rows(); row++)
    {
        for(int col = 0; col != Y_Train.cols(); col++)
        {
            std::cout << std::to_string(Y_Train(row,col)) << std::endl;
        }
    }


    return 0;
}
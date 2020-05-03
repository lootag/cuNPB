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
    float alpha = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.999;
    
    char *directory = "/content/data"; 
    int ret;
    ret = chdir(directory);
    Eigen::MatrixXf X_Train = from_csv(train, train_rows, train_cols);
    Eigen::MatrixXf X_Test = from_csv(test, test_rows, train_cols);
    Eigen::MatrixXf Y_Train = from_csv(labels, train_rows, labels_cols);
    std::string mean = "mean.csv";
    std::string variance = "variance.csv";

    model Model(X_Train, Y_Train, X_Test, alpha, beta1, beta2);
    Model.Train();
    Model.PredictMean();
    to_csv(Model.get_mu(), mean); 

    Model.PredictVariance();
    to_csv(Model.get_Sigma_2(), variance);
    
    return 0;
}
#ifndef __MODEL_H__
#define __MODEL_H__
#include <eigen3/Eigen/Dense>
#include <vector>

class model
{
private:
    float sigma;
    float l;
    float alpha;
    float beta1;
    float beta2;
    Eigen::MatrixXf X_Train;
    Eigen::MatrixXf Y_Train;
    Eigen::MatrixXf X_Test;
    Eigen::MatrixXf mu;
    Eigen::MatrixXf Sigma_2;
    void set_mu(Eigen::MatrixXf _mu);
    void set_Sigma_2(Eigen::MatrixXf _Sigma_2);
    float get_alpha();
    float get_beta1();
    float get_beta2();
    float get_sigma();
    float get_l();
    void set_sigma(float _sigma);
    void set_l(float _l);
    Eigen::MatrixXf get_X_Train();
    Eigen::MatrixXf get_Y_Train();
    Eigen::MatrixXf get_X_Test();
    
public:
    model(Eigen::MatrixXf _X_Train, Eigen::MatrixXf _Y_Train, Eigen::MatrixXf _X_Test, float _alpha, float _beta1, float _beta2);
    ~model();
    Eigen::MatrixXf get_mu();
    Eigen::MatrixXf get_Sigma_2();
    void Train();
    void PredictMean();
    void PredictVariance(); 
};

inline model::model(Eigen::MatrixXf _X_Train, Eigen::MatrixXf _Y_Train, Eigen::MatrixXf _X_Test, float _alpha, float _beta1, float _beta2)
{
    X_Train = _X_Train;
    Y_Train = _Y_Train;
    X_Test = _X_Test;
    alpha = _alpha;
    beta1 = _beta1;
    beta2 = _beta2; 
}

inline Eigen::MatrixXf model::get_mu()
{
    return mu;
}

inline Eigen::MatrixXf model::get_Sigma_2()
{
    return Sigma_2; 
}

inline void model::set_mu(Eigen::MatrixXf _mu)
{
    mu = _mu;
}

inline void model::set_Sigma_2(Eigen::MatrixXf _Sigma_2)
{
    Sigma_2 = _Sigma_2;
}

inline float model::get_sigma()
{
    return sigma;
}

inline float model::get_l()
{
    return l;
}

inline void model::set_sigma(float _sigma)
{
    sigma = _sigma;
}

inline void model::set_l(float _l)
{
    l = _l;
}

inline float model::get_alpha()
{
    return alpha;
}

inline float model::get_beta1()
{
    return beta1;
}

inline float model::get_beta2()
{
    return beta2;
}

inline Eigen::MatrixXf model::get_X_Train()
{
    return X_Train;
}

inline Eigen::MatrixXf model::get_Y_Train()
{
    return Y_Train;
}

inline Eigen::MatrixXf model::get_X_Test()
{
    return X_Test;
}

inline model::~model()
{
}


#endif
#ifndef __MODEL_H__
#define __MODEL_H__
#include <eigen3/Eigen/Dense>
#include <vector>

class model
{
private:
    float sigma_2;
    float l_2;
    Eigen::MatrixXf X_Train;
    Eigen::MatrixXf Y_Train;
    Eigen::MatrixXf mu;
    Eigen::MatrixXf Sigma_2;
    void set_mu(Eigen::MatrixXf _mu);
    void set_Sigma_2(Eigen::MatrixXf _Sigma_2);
    float get_sigma_2();
    float get_l_2();
    void set_sigma_2(float _sigma_2);
    void set_l_2(float _l_2);
    
public:
    model(Eigen::MatrixXf X_Train, Eigen::MatrixXf Y_Train);
    ~model();
    Eigen::MatrixXf get_mu();
    Eigen::MatrixXf get_Sigma_2();
    std::vector<float> Train();
};

inline model::model(Eigen::MatrixXf _X_Train, Eigen::MatrixXf _Y_Train)
{
    X_Train = _X_Train;
    Y_Train = _Y_Train;
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

inline float model::get_sigma_2()
{
    return sigma_2;
}

inline float model::get_l_2()
{
    return l_2;
}

inline void model::set_sigma_2(float _sigma_2)
{
    sigma_2 = _sigma_2;
}

inline void model::set_l_2(float _l_2)
{
    l_2 = _l_2;
}

inline model::~model()
{
}


#endif
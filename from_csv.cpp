#include <iostream>
#include <fstream>
#include <regex>
#include <eigen3/Eigen/Dense>
#include "preprocessing.h"

using namespace Eigen;

MatrixXf from_csv(std::string path, int rows, int cols)
{
    std::ifstream inFile;
    inFile.open(path);
    std::string currentLine; 
    std::vector<std::string> allLines;
    std::regex findNumbers(R"(\d+(\.\d+)*)");
    std::smatch numbers;
    MatrixXf toReturn(rows, cols);
    
    
    if(!inFile)
    {
        std::cout << "Sorry, I couldn't open the file. Now I'll die :(" << std::endl;
        exit(1);
    }
    
    while(inFile.good())
    {
        try
        {
            getline(inFile, currentLine);
            allLines.push_back(currentLine);
        }
        catch(std::exception& e)
        {
            std::cout << "Successfully read data" << std::endl;
        }
        
    }


    
    for(int i = 0; i != allLines.size() ; i++)
    {
        std::string toProcess(allLines[i]);
        std::vector<std::string> matched;
        while(std::regex_search(toProcess, numbers, findNumbers))
        {
            matched.push_back(numbers.str(0));
            toProcess = numbers.suffix().str();
        }
        
        for(int j = 0; j != matched.size() ; j++)
        {
        
            toReturn(i,j) = std::stof(matched[j]);
        }
    }
    
    return toReturn;
    
    
    
    
}
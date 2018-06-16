#pragma once
//#include "Eigen/Core"
#include<string>
#include <iostream>
#include<vector>
#include "tiny_dnn/tiny_dnn.h"
using namespace tiny_dnn;
//using namespace Eigen;
void Read_File(float *Indata, int size,const char *FileName);
void Read_File(float *Indata, int size, std::string  FileName);

void Read_File(tiny_dnn::vec_t &Indata, const char *FileName);

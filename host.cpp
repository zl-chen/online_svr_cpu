#include <stdio.h>
#include <fstream>
#include <time.h>
#include <vector>
#include <math.h>
#include <iostream>
#include <string>
#include "onlinesvr.h"

#define NUM_ROWS 1022
#define NUM_FEATURES 3

int N;
int *A,*B,*C;


using namespace std;

const int msg_len = 1024;
char msg[msg_len];

void cleanup();

void load_data(double (*X)[NUM_FEATURES],double *Y){

    // 导入
    ifstream ifs("/home/shu_students/czl/online_svr_cpu/data/data.csv",ifstream::in);

    char line[1024];
    ifs.getline(line,1024);

    int idx = 0;
    
    while(!ifs.eof()){
        char temp;
        for(int i=0;i<3;++i){
            ifs>>X[idx][i];
            ifs>>temp;
        }
        ifs>>Y[idx];
        ++idx;
    }
    ifs.close();

    // 数据清洗
    vector<double> means;
    for(int i=0;i<NUM_FEATURES;++i){
        double mean = 0;
        for(int j=0;j<NUM_ROWS;++j){
            mean += X[j][i];
        }
        mean = mean/NUM_ROWS;

        means.push_back(mean);
    }
    vector<double> sd_vec;
    for(int i=0;i<NUM_FEATURES;++i){
        double sd = 0;
        
        for(int j=0;j<NUM_ROWS;++j){
            sd += (X[j][i] - means[i]) * (X[j][i] - means[i]);
        }

        sd = sqrt(sd/NUM_ROWS);

        sd_vec.push_back(sd);
    }



    for(int i=0;i<NUM_FEATURES;++i){
        double mean = means[i];
        double sd = sd_vec[i];

        for(int j=0;j<NUM_ROWS;++j){
            X[j][i] = (X[j][i] - mean)/sd;
        }
    }

    

}


void train(){
    double X[NUM_ROWS][NUM_FEATURES];
    double Y[NUM_ROWS];
    load_data(X,Y);

    OnlineSVR online_svr(3,143,0.1,0.1,0.5);

    int train_num = 512;

    clock_t start = clock();
    
    for(int i=0;i<train_num;++i){
   
        vector<double> xVec(begin(X[i]),end(X[i]));
        online_svr.learn(xVec,Y[i]);

        
         vector<vector<double>> newX;
        vector<double> xVec2(begin(X[i+1]),end(X[i+1]));
        newX.push_back(xVec);

        cout << i << "   " << online_svr.predict(newX)[0] << endl ;

    }

    clock_t end = clock();
    cout << double(end-start)/CLOCKS_PER_SEC << "s" << endl;

    /*
    double mse = 0.0;
    for(int i=train_num;i<train_num+100;++i){
        vector<vector<double>> newX;
        vector<double> xVec(begin(X[i]),end(X[i]));
        newX.push_back(xVec);

        double y_pre = online_svr.predict(newX)[0];

        mse += (y_pre-Y[i])*(y_pre-Y[i]);

    }


    cout << mse/100 << endl;
    */
}

void matMul(){

    for(int i=0;i<N;++i){
        for(int j=0;j<N;++j){
            int result = 0;

            for(int k=0;k<N;++k){
                result += A[i * N + k] * B[k * N + j];
            }

            C[i * N + j] = result;
        }
    }

}


int main(int argc ,char** argv)
{

   



    train();
}


void cleanup(){
    return ;
}
/*
* Copyright (c) 2021, Stefan Güttel, Xinye Chen
* Licensed under BSD 3-Clause License
* All rights reserved.
*

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <iostream>
#include <vector>
#include <bits/stdc++.h>
using namespace std;

template<typename T>
T dist_manhattan(T value1, T value2){
    return abs(value1 - value2);
}

template<typename T>
T dist_euclidean(T value1, T value2){
    return pow((value1 - value2),2);
}
    // auto dist = [](auto a, auto b) { return abs(a - b);};
    // auto dist = [](auto a, auto b) { return pow((a - b),2);};

template<typename T>
float DTW(std::vector<T> const &series1, std::vector<T> const &series2, float (*dist)(float, float), bool i_sqrt){
    int nrows(series1.size()), ncols(series2.size());

    float distance_matrix[nrows + 1][ncols + 1];
    
    for (int i=0; i<nrows+1; i++){
        for (int j=0; j<nrows+1; j++){
            distance_matrix[i][j] = std::numeric_limits<float>::infinity();
        }
    }
    distance_matrix[0][0] = 0.0;
    float store_elements[3];
    float cost, min_last;
    for (int i=1; i<nrows+1; i++){
        for (int j=1; j<nrows+1; j++){
            cost = dist((float)series1[i-1], (float)series2[j-1]);
            store_elements[0] = distance_matrix[i-1][j];
            store_elements[1] = distance_matrix[i][j-1];
            store_elements[2] =  distance_matrix[i-1][j-1];
            min_last = *min_element(store_elements, store_elements+3);
            distance_matrix[i][j] = cost + min_last;
            // cout << distance_matrix[i][j] << " ";
        }
        // cout << endl;
    }
    
    if (i_sqrt == true){
        return sqrt(distance_matrix[series1.size()][series2.size()]);
    }else{
        return distance_matrix[series1.size()][series2.size()];
    }
}


int main(){
    std::vector<double> ts1 = {2,3,3,2,3,4,2,1,3,2,9.234,9.12312, 9.02,5.2};
    std::vector<double> ts2 = {1.2,2,1,2,3.5,4.1,1.8,2.5,3,2, 3,2,9.234};
    cout << "manhattan: " << DTW(ts1, ts2, dist_manhattan, 0) << endl;
    cout << "euclidean: " << DTW(ts1, ts2, dist_euclidean, 1) << endl;
    return 0;
}
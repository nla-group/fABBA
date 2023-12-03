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


#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H

#include <vector>
#include <string>
#include <cmath>
#include "global_func.h"
#include "digitization.h"
#include "operator_loader.h"



template <typename T> std::vector<std::vector<double> > inverse_digitize(Model&);
template <typename T> std::vector<std::vector<T> > quantize(std::vector<std::vector<T> >);
template <typename T> std::vector<T> inverse_compress(std::vector<std::vector<T> >&, double& start=0);



std::vector<std::vector<double> > inverse_digitize(Model& parameters){
    std::vector<std::vector<double> > pieces(0);
    for (auto & symbol : parameters._symbols){
        pieces.push_back(parameters._centers[parameters._hashmap[symbol]]); //std::string(1, symbol)
    }
    return pieces;
}



template <typename T>
std::vector<std::vector<T> > quantize(std::vector<std::vector<T> > pieces){
    auto num = (int)pieces.size();
    double corr;
    if (num == 1){
        pieces[0][0] = round(pieces[0][0]);
    }else{
        for(int p=0 ; p<num-1; p++){
            corr = round(pieces[p][0]) - pieces[p][0];
            pieces[p][0] = round(pieces[p][0] + corr);
            pieces[p+1][0] = pieces[p+1][0] - corr;
            if (pieces[p][0] == 0){
                pieces[p][0] = 1;
                pieces[p+1][0] = pieces[p+1][0] - 1;
            }
        }
        pieces[num-1][0] = round(pieces[num-1][0]);
    }
    return pieces;
}



template <typename T>
std::vector<T> stable_range(T n){
    std::vector<T> inc_range;
    for(int i=0; i<n; i++){
        inc_range.push_back(i);
    }
    return inc_range;
}



template <typename T>
std::vector<T> inverse_compress(std::vector<std::vector<T> >& pieces, double& start){
    std::vector<T> series = {start}, x, y, srange;
    
    double scale;
    for(int i=0; i < pieces.size(); i++){
        scale = pieces[i][1] / pieces[i][0];
        srange = stable_range(pieces[i][0] + 1.0); 
        x = scale * srange;
        y = series[series.size()-1] + x; //*(series.end()-1) + x;
        series.insert(series.end(), y.begin()+1, y.end()); 
    }
    return series;
}

#endif // RECONSTRUCTION_H
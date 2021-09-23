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


#ifndef TOYDATA_H
#define TOYDATA_H

#include <random>   
#include <algorithm>  // generate
#include <vector>     // vector
#include <iterator>   // iterator
#include <iostream>   // cout
#include <string>

std::vector<double> generate_random_sequence(int, std::string);

/* possible other distributions
   - uniform_real_distribution
   - to be continued
*/

std::default_random_engine generator;
std::normal_distribution<double> Normaldistribution(0.0,1.0);

double normal_dist(){
  return Normaldistribution(generator);
}

std::uniform_real_distribution<double> Uniformdistribution(0.0,1.0);
double uniform_dist(){
  return Uniformdistribution(generator);
}

const double NUMBER = 1.0;
double trivial() {return NUMBER;}

std::vector<double> generate_random_sequence(int num, std::string distribution){
  std::vector<double> random_sequence(num);
  
  if (distribution == "normal"){
      std::generate(random_sequence.begin(), random_sequence.end(), 
             normal_dist);
    }else if (distribution == "uniform"){
        std::generate(random_sequence.begin(), random_sequence.end(),
             uniform_dist);
    }else{
        std::generate(random_sequence.begin(), random_sequence.end(), 
             trivial);
    }
  return random_sequence;
}


#endif // TOYDATA_H
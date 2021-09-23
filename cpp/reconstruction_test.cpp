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
#include <string>
#include <vector>
#include <algorithm>

#include "compress.h"
#include "digitization.h"
#include "reconstruction.h"
#include "ABBA.h"
using namespace std;


int main(){
    vector<double> test_series = {1, 2, -3, 2, 3, -5};
    cout << "*****************************test function" << endl<< endl; 
    vector<vector<double> > pieces = Compress(test_series, 0.5, 10000, true);
    Model model = Digitize(pieces, 0.5, "2-norm", 1);
    print_matrix("pieces: ", pieces);
    print_vector("symbols: ", model._symbols); 
    vector<vector<double> > r_pieces = inverse_digitize(model);
    print_matrix("reconstruction pieces: ", r_pieces);
    vector<vector<double> > q_pieces = quantize(r_pieces);
    print_matrix("quantize pieces: ", q_pieces);
    vector<double> r_series = inverse_compress(q_pieces, test_series[0]);
    print_vector("reconstruction series: ", r_series);
}

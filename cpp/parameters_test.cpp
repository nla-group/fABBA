
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
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include "digitization.h"
#include "operator_loader.h"
#include "global_func.h"


int main(){
    std::vector<std::vector<double> > data = {{2,0.3}, {2,0.2}, {1,0.9}, {3, -0.24}, {3, -0.12}, {3, 0.67}};
    Model parameters;
    print_matrix(data);
    data = Scale_pieces(data, parameters);
    print_matrix(data);
    std::vector<std::size_t> test_v= {1,2,3,5,5,5};
    
    if (true) print_vector(vslice(test_v, 2, test_v.size()));

    std::string sorting = "2-norm";
    if (sorting == "2-norm"){
        std::cout << "test digitization:" << std::endl;  
    }
    int scl = 1;
    double alpha = 0.5;
    parameters = Digitize(data, alpha, "2-norm", scl);
    print_vector(parameters._symbols);

    std::cout << std::round(3.212) << std::endl;
}

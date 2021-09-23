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

* This program design for reading time series storing in csv file.
* save file:
*          std::vector<std::pair<std::string, std::vector<double>>> vals = {{"column 1": series1}, {"column 2": series2}};
*          data_to_csv(vals, "test.csv",  false); // false means do not store the index
* read file: 
*          std::vector<std::pair<std::string, std::vector<std::string>>> pairs= read_csv_to_pairs("test.csv", true); // true means read the first column.
*/

#ifndef CSV_READER_H
#define CSV_READER_H

#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <sstream> 
#include <numeric>
#include <utility> 


template<typename T>
void data_to_csv_base(std::vector<std::pair<std::string, std::vector<T>>>& pairs, std::string filename){
    std::ofstream write_file(filename);
    for(int j = 0; j < pairs.size(); ++j){
        write_file << pairs.at(j).first;
        if(j != pairs.size() - 1) write_file << ",";
    }
    write_file << "\n";

    for(int i = 0; i < pairs.at(0).second.size(); ++i){
        for(int j = 0; j < pairs.size(); ++j){
            write_file << pairs.at(j).second.at(i);
            if(j != pairs.size() - 1) write_file << ",";
        }
        write_file << "\n";
    }
    write_file.close();
}


template<typename T>
void data_to_csv(std::vector<std::pair<std::string, std::vector<T>>>& pairs, std::string filename, bool index){
    T NUM = pairs.at(0).second.size();
    std::vector<std::pair<std::string, std::vector<T>>> construct_pairs;

    if (index){
        std::vector<T> range(NUM);
        std::generate(range.begin(), range.end(), [n = 0] () mutable { return n++; });

        std::pair<std::string, std::vector<T>> fpair = {"Index", range};
        construct_pairs.push_back(fpair);
        for (auto & pair : pairs){
            construct_pairs.push_back({pair});
        }
    }else{
        for (auto & pair : pairs){
            construct_pairs.push_back({pair});
        }
    }
    data_to_csv_base(construct_pairs, filename);
}


std::vector<std::pair<std::string, std::vector<std::string>>> read_csv_to_pairs(std::string filename, bool index=true){
    std::vector<std::pair<std::string, std::vector<std::string>>> pairs;
    std::ifstream read_file(filename);
    try
    {
        if(!read_file.is_open()) throw std::runtime_error("runtime error! ");
        std::string line, colname;
        std::string val;
        if(read_file.good()){
            std::getline(read_file, line);
            std::stringstream ss(line);

            while(std::getline(ss, colname, ',')) pairs.push_back({colname, std::vector<std::string> {}});
        }

        while(std::getline(read_file, line)){
            std::stringstream ss(line);
            int index_column(1);
            if (index) index_column = 0;

            while(ss >> val){
                pairs.at(index_column).second.push_back(val);
                if(ss.peek() == ',') ss.ignore();
                index_column = index_column + 1;
            }
        }
        read_file.close();
        return pairs;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

#endif CSV_READER_H
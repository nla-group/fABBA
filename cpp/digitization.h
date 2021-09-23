/*
* Copyright (c) 2021, Stefan Güttel, Xinye Chen
* Licensed under BSD 3-Clause License
* you may not use this file except in compliance with the License. 
*
* Approximate a time series using a continuous piecewise linear function.
* Construct 2D vector compression pieces.
*
* Greedy 2D array, using tolernce alpha and len/inc scaling parameter scl.
* A 'temporary' group center, which we call it starting point,
* is used  when assigning pieces to clusters. This temporary
* cluster is the first piece available after appropriate scaling 
* and sorting of all pieces. After finishing the grouping procedure,
* the centers are calculated the mean value of the objects within the 
* clusters
*/

#ifndef DIGITIZATION_H
#define DIGITIZATION_H

#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

#include "operator_loader.h"
#include "global_func.h"


struct Model{
    int _scl = 1;
    double _len_Param[2] = {0.0, 0.0};
    double _inc_Param[2] = {0.0, 0.0};
    std::vector<std::string> _symbols;
    std::map<std::string, int> _hashmap;
    std::map<int, std::string> _inverse_hashmap;
    std::vector<std::vector<double> > _centers;
};


template <typename T> void Cal_standard_value(std::vector<T>,  double &, double &);
template <typename T> std::vector<std::vector<T> > Scale_pieces(std::vector<std::vector<T> >, Model &, int scl=1);
template <typename T> std::vector<int> Aggregate(std::vector<std::vector<T> >, Model &, double alpha=0.5, std::string sorting="2-norm", bool verbose=true);
template <typename T> Model Digitize(std::vector<std::vector<T> >, double alpha=0.5, std::string sorting="2-norm", int scl=1, bool verbose=true);


template <typename T>
std::vector<int> Aggregate(std::vector<std::vector<T> > pieces, Model &parameters, double alpha, std::string sorting, bool verbose){
    auto num = (double) pieces.size();
    std::vector<T> indices = arange(num);
    // std::cout << "pieces" << std::endl;
    // print_matrix(pieces);
    if (sorting == "lexi"){
        indices = arg_lexisort(pieces);
    }else if (sorting == "2-norm"){
        indices = argsort_norm(pieces, sorting);
    }else if (sorting == "1-norm"){
        indices = argsort_norm(pieces, sorting);
    }

    int lab = 0;
    std::size_t starting_point;
    std::vector<int> labels;
    labels.insert(labels.end(), num, -1);
    std::vector<std::vector<T> > splist(0); // storing starting point information
    double center_norm, distance; 
    std::vector<double> clustc, center;
    std::string symbol;
    std::vector<std::vector<T> > cpieces_stack(0), centers(0);
    double len_m(0.0), inc_m(0.0);
    for (std::size_t i=0; i<num; i++){
        starting_point = indices[i];
        
        if (labels[starting_point] >= 0){
            continue;
        }else{
            cpieces_stack.clear();
            clustc = pieces[starting_point]; // starting point as temporary center
            cpieces_stack.push_back(clustc);

            labels[starting_point] = lab;
            std::vector<double> insert_sp = {(double)starting_point, (double)lab, clustc[0], clustc[1]};
            splist.push_back(insert_sp);

            if(sorting == "2-norm"){
                center_norm = norm(clustc);
            }else if (sorting == "1-norm"){
                center_norm = clustc[0] + abs(clustc[1]);
            }
        }

        for (auto &j: vslice(indices, i, indices.size())){
            if (labels[j] >= 0) continue;
            // early stopping
            if (sorting == "lexi"){
                if ((pieces[j][0] - pieces[starting_point][0]) > alpha) break;
            }else if (sorting == "2-norm"){
                if (norm(pieces[j]) - center_norm > alpha) break;
            }else if (sorting == "1-norm"){
                if (0.707101 * (pieces[j][0] + abs(pieces[j][1]) - center_norm) > alpha) break;
            }

            distance = norm(clustc - pieces[j]);
            if (distance <= alpha){
                labels[j] = lab;
                cpieces_stack.push_back(pieces[j]);
            }
        }

        symbol = generate_symbol(33 + lab);// ASCII convert *static_cast<unsigned char>
        parameters._hashmap[symbol] = lab; 
        parameters._inverse_hashmap[lab] = symbol; 

        // calculate centers
        for (std::size_t c = 0; c < cpieces_stack.size(); c++){
            len_m = len_m + cpieces_stack[c][0];
            inc_m = inc_m + cpieces_stack[c][1];
        }

        len_m = len_m / cpieces_stack.size();
        inc_m = inc_m / cpieces_stack.size();

        len_m =  parameters._len_Param[1] * len_m + parameters._len_Param[0];
        inc_m =  parameters._inc_Param[1] * inc_m + parameters._inc_Param[0];
        
        center = {len_m, inc_m}; 
        centers.push_back(center);

        len_m = 0.0; inc_m = 0.0;
        lab = lab + 1; 
    }

    if (verbose == true){
        std::cout << "Digitization: Reduced pieces of length " << pieces.size() << " to " << lab << " symbols" << std::endl;
    }
    parameters._centers = centers;
    return labels;
}



template <typename T>
Model Digitize(std::vector<std::vector<T> > pieces, double alpha, std::string sorting, int scl, bool verbose){
    Model parameters;
    pieces = remove_col(pieces,2);
    std::vector<std::vector<T> > spieces = Scale_pieces(pieces, parameters, scl);
    std::vector<int> labels = Aggregate(spieces, parameters, alpha, sorting, verbose);
    //double label_max = *std::max_element(labels.begin(), labels.end());
    std::vector<std::string> symbols;
    for (int &label : labels){
        symbols.push_back(parameters._inverse_hashmap[label]);
    } 
    parameters._symbols = symbols;
    return parameters;
}



template <typename T>
std::vector<std::vector<T> > Scale_pieces(std::vector<std::vector<T> > pieces, Model &parameters, int scl){
    /*
        Scale the pieces, speed up the aggregation, and reduce the influence of length over increment.
    */
    int num = pieces.size();
    std::vector<T> lens(num); 
    std::vector<T> incs(num); 
    for (std::size_t i=0; i < num; i++){
        lens[i] = pieces[i][0];
        incs[i] = pieces[i][1];
    }
    double len_m(0.0), inc_m(0.0), len_std(0.0), inc_std(0.0);
    Cal_standard_value(lens, len_m, len_std);
    Cal_standard_value(incs, inc_m, inc_std);

    if (len_std != 0){
        for (std::size_t i=0; i < num; i++){
            pieces[i][0] = pieces[i][0] * scl  / len_std;
            pieces[i][1] = pieces[i][1] / inc_std;
        }
        // parameters._len_Param[0] = len_m;
        parameters._len_Param[1] = len_std;
        // parameters._inc_Param[0] = inc_m;
        parameters._inc_Param[1] = inc_std;

    }else{
        for (std::size_t i=0; i < num; i++){
            pieces[i][0] = pieces[i][0] * scl;
            pieces[i][1] = pieces[i][1] / inc_std;
        }
        // parameters._len_Param[0] = len_m;
        parameters._len_Param[1] = 1;
        // parameters._inc_Param[0] = inc_m;
        parameters._inc_Param[1] = inc_std;
    }

    return pieces;
}



template <typename T>
void Cal_standard_value(std::vector<T> vec, double &mean, double &standard_deviation){
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    mean = sum / vec.size();
    std::vector<double> diff(vec.size());
    std::transform(vec.begin(), vec.end(), diff.begin(), std::bind2nd(std::minus<double>(), mean));
    double square_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    standard_deviation = std::sqrt(square_sum / vec.size());
}



#endif // DIGITIZATION_H

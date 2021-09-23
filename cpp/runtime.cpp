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

* Apply fABBA to 20 noise series, evaluate the fABBA performance on these benchmarks 
* The experimental results can be compared with the fABBA python implementation.
*/

#include <iostream>
#include <vector>
#include <chrono>
#include "toydata.h"
#include "csv_reader.h"
#include "ABBA.h"
//#include "dynamicTimeWarping.h"
using namespace std;

int main(){
    size_t series_size = 10000; // set 1000 if measured by DTW
    chrono::steady_clock time_point;
    double tol(0.01), alpha(0.01);
    vector<string> symbols;
    vector<double> r_fabba_series, errors, errors_dtw, runtimes;
    vector<pair<string, vector<double>>> benchmarks={}; 
    pair<string, vector<double>> test; 


    auto ts = generate_random_sequence(series_size, "uniform");
    cout << "\n*****************************test 1" << endl; 
    // print_vector(ts);
    auto start = time_point.now();
    fABBA::ABBA fabba = fABBA::ABBA(tol, alpha, "lexi", 1, series_size, true);
    symbols= fabba.fit_transform(ts);
    r_fabba_series = fabba.inverse_transform(ts[0]);
    auto end = time_point.now();
    auto runtime = static_cast<chrono::duration<double>>(end - start);
    
    //cout << symbols << endl;
    double error = norm(ts - r_fabba_series);
    //double dtw =  DTW(ts, r_fabba_series, dist_euclidean, 1);
    errors.push_back(error);
    //errors_dtw.push_back(dtw);

    runtimes.push_back(runtime.count());
    cout << "runtime: " << runtime.count()<<" seconds" << endl;
    //cout << "error:" << error << endl; 
    test = {"test 1", ts};
    benchmarks.push_back(test); 
    test = {"reconst test 1", r_fabba_series};
    benchmarks.push_back(test); 


    ts = generate_random_sequence(series_size, "uniform");
    cout << "\n*****************************test 2" << endl; 
    // print_vector(ts);
    start = time_point.now();
    fabba = fABBA::ABBA(tol, alpha, "lexi", 1, series_size, true);
    symbols= fabba.fit_transform(ts);
    r_fabba_series = fabba.inverse_transform(ts[0]);
    end = time_point.now();

    // cout << symbols << endl;
    runtime = static_cast<chrono::duration<double>>(end - start);
    error = norm(ts - r_fabba_series);
    //dtw =  DTW(ts, r_fabba_series, dist_euclidean, 1);
    errors.push_back(error);
    //errors_dtw.push_back(dtw);

    runtimes.push_back(runtime.count());
    cout << "runtime: " << runtime.count()<<" seconds" << endl;
    //cout << "error:" << error << endl; 
    test = {"test 2", ts};
    benchmarks.push_back(test); 
    test = {"reconst test 2", r_fabba_series};
    benchmarks.push_back(test); 


    ts = generate_random_sequence(series_size, "uniform");
    cout << "\n*****************************test 3" << endl; 
    // print_vector(ts);
    start = time_point.now();
    fabba = fABBA::ABBA(tol, alpha, "lexi", 1, series_size, true);
    symbols= fabba.fit_transform(ts);
    r_fabba_series = fabba.inverse_transform(ts[0]);
    end = time_point.now();

    // cout << symbols << endl;
    runtime = static_cast<chrono::duration<double>>(end - start);
    error = norm(ts - r_fabba_series);
    //dtw =  DTW(ts, r_fabba_series, dist_euclidean, 1);
    errors.push_back(error);
    //errors_dtw.push_back(dtw);

    runtimes.push_back(runtime.count());
    cout << "runtime: " << runtime.count()<<" seconds" << endl;
    //cout << "error:" << error << endl; 
    test = {"test 3", ts};
    benchmarks.push_back(test); 
    test = {"reconst test 3", r_fabba_series};
    benchmarks.push_back(test); 


    ts = generate_random_sequence(series_size, "uniform");
    cout << "\n*****************************test 4" << endl; 
    // print_vector(ts);
    start = time_point.now();
    fabba = fABBA::ABBA(tol, alpha, "lexi", 1, series_size, true);
    symbols= fabba.fit_transform(ts);
    r_fabba_series = fabba.inverse_transform(ts[0]);
    end = time_point.now();

    // cout << symbols << endl;
    runtime = static_cast<chrono::duration<double>>(end - start);
    error = norm(ts - r_fabba_series);
    //dtw =  DTW(ts, r_fabba_series, dist_euclidean, 1);
    errors.push_back(error);
    //errors_dtw.push_back(dtw);

    runtimes.push_back(runtime.count());
    cout << "runtime: " << runtime.count()<<" seconds" << endl;
    //cout << "error:" << error << endl; 
    test = {"test 4", ts};
    benchmarks.push_back(test); 
    test = {"reconst test 4", r_fabba_series};
    benchmarks.push_back(test); 


    ts = generate_random_sequence(series_size, "uniform");
    cout << "\n*****************************test 5" << endl; 
    // print_vector(ts);
    start = time_point.now();
    fabba = fABBA::ABBA(tol, alpha, "lexi", 1, series_size, true);
    symbols= fabba.fit_transform(ts);
    r_fabba_series = fabba.inverse_transform(ts[0]);
    end = time_point.now();

    // cout << symbols << endl;
    runtime = static_cast<chrono::duration<double>>(end - start);
    error = norm(ts - r_fabba_series);
    //dtw =  DTW(ts, r_fabba_series, dist_euclidean, 1);
    errors.push_back(error);
    //errors_dtw.push_back(dtw);

    runtimes.push_back(runtime.count());
    cout << "runtime: " << runtime.count()<<" seconds" << endl;
    //cout << "error:" << error << endl; 
    test = {"test 5", ts};
    benchmarks.push_back(test); 
    test = {"reconst test 5", r_fabba_series};
    benchmarks.push_back(test); 


    ts = generate_random_sequence(series_size, "uniform");
    cout << "\n*****************************test 6" << endl; 
    // print_vector(ts);
    start = time_point.now();
    fabba = fABBA::ABBA(tol, alpha, "lexi", 1, series_size, true);
    symbols= fabba.fit_transform(ts);
    r_fabba_series = fabba.inverse_transform(ts[0]);
    end = time_point.now();

    // cout << symbols << endl;
    runtime = static_cast<chrono::duration<double>>(end - start);
    error = norm(ts - r_fabba_series);
    //dtw =  DTW(ts, r_fabba_series, dist_euclidean, 1);
    errors.push_back(error);
    //errors_dtw.push_back(dtw);

    runtimes.push_back(runtime.count());
    cout << "runtime: " << runtime.count()<<" seconds" << endl;
    //cout << "error:" << error << endl; 
    test = {"test 6", ts};
    benchmarks.push_back(test); 
    test = {"reconst test 6", r_fabba_series};
    benchmarks.push_back(test); 


    ts = generate_random_sequence(series_size, "uniform");
    cout << "\n*****************************test 7" << endl; 
    // print_vector(ts);
    start = time_point.now();
    fabba = fABBA::ABBA(tol, alpha, "lexi", 1, series_size, true);
    symbols= fabba.fit_transform(ts);
    r_fabba_series = fabba.inverse_transform(ts[0]);
    end = time_point.now();

    // cout << symbols << endl;
    runtime = static_cast<chrono::duration<double>>(end - start);
    error = norm(ts - r_fabba_series);
    //dtw =  DTW(ts, r_fabba_series, dist_euclidean, 1);
    errors.push_back(error);
    //errors_dtw.push_back(dtw);

    runtimes.push_back(runtime.count());
    cout << "runtime: " << runtime.count()<<" seconds" << endl;
    //cout << "error:" << error << endl; 
    test = {"test 7", ts};
    benchmarks.push_back(test); 
    test = {"reconst test 7", r_fabba_series};
    benchmarks.push_back(test); 
    

    ts = generate_random_sequence(series_size, "uniform");
    cout << "\n*****************************test 8" << endl; 
    // print_vector(ts);
    start = time_point.now();
    fabba = fABBA::ABBA(tol, alpha, "lexi", 1, series_size, true);
    symbols= fabba.fit_transform(ts);
    r_fabba_series = fabba.inverse_transform(ts[0]);
    end = time_point.now();

    // cout << symbols << endl;
    runtime = static_cast<chrono::duration<double>>(end - start);
    error = norm(ts - r_fabba_series);
    //dtw =  DTW(ts, r_fabba_series, dist_euclidean, 1);
    errors.push_back(error);
    //errors_dtw.push_back(dtw);

    runtimes.push_back(runtime.count());
    cout << "runtime: " << runtime.count()<<" seconds" << endl;
    //cout << "error:" << error << endl; 
    test = {"test 8", ts};
    benchmarks.push_back(test); 
    test = {"reconst test 8", r_fabba_series};
    benchmarks.push_back(test); 


    ts = generate_random_sequence(series_size, "uniform");
    cout << "\n*****************************test 9" << endl; 
    // print_vector(ts);
    start = time_point.now();
    fabba = fABBA::ABBA(tol, alpha, "lexi", 1, series_size, true);
    symbols= fabba.fit_transform(ts);
    r_fabba_series = fabba.inverse_transform(ts[0]);
    end = time_point.now();

    // cout << symbols << endl;
    runtime = static_cast<chrono::duration<double>>(end - start);
    error = norm(ts - r_fabba_series);
    //dtw =  DTW(ts, r_fabba_series, dist_euclidean, 1);
    errors.push_back(error);
    //errors_dtw.push_back(dtw);

    runtimes.push_back(runtime.count());
    cout << "runtime: " << runtime.count()<<" seconds" << endl;
    //cout << "error:" << error << endl; 
    test = {"test 9", ts};
    benchmarks.push_back(test); 
    test = {"reconst test 9", r_fabba_series};
    benchmarks.push_back(test); 


    ts = generate_random_sequence(series_size, "uniform");
    cout << "\n*****************************test 10" << endl; 
    // print_vector(ts);
    start = time_point.now();
    fabba = fABBA::ABBA(tol, alpha, "lexi", 1, series_size, true);
    symbols= fabba.fit_transform(ts);
    r_fabba_series = fabba.inverse_transform(ts[0]);
    end = time_point.now();

    // cout << symbols << endl;
    runtime = static_cast<chrono::duration<double>>(end - start);
    error = norm(ts - r_fabba_series);
    //dtw =  DTW(ts, r_fabba_series, dist_euclidean, 1);
    errors.push_back(error);
    //errors_dtw.push_back(dtw);

    runtimes.push_back(runtime.count());
    cout << "runtime: " << runtime.count()<<" seconds" << endl;
    //cout << "error:" << error << endl; 
    test = {"test 10", ts};
    benchmarks.push_back(test); 
    test = {"reconst test 10", r_fabba_series};
    benchmarks.push_back(test); 


    ts = generate_random_sequence(series_size, "normal");
    cout << "\n*****************************test 11" << endl; 
    // print_vector(ts);
    start = time_point.now();
    fabba = fABBA::ABBA(tol, alpha, "lexi", 1, series_size, true);
    symbols= fabba.fit_transform(ts);
    r_fabba_series = fabba.inverse_transform(ts[0]);
    end = time_point.now();

    // cout << symbols << endl;
    runtime = static_cast<chrono::duration<double>>(end - start);
    error = norm(ts - r_fabba_series);
    //dtw =  DTW(ts, r_fabba_series, dist_euclidean, 1);
    errors.push_back(error);
    //errors_dtw.push_back(dtw);

    runtimes.push_back(runtime.count());
    cout << "runtime: " << runtime.count()<<" seconds" << endl;
    //cout << "error:" << error << endl; 
    test = {"test 11", ts};
    benchmarks.push_back(test); 
    test = {"reconst test 11", r_fabba_series};
    benchmarks.push_back(test); 


    ts = generate_random_sequence(series_size, "normal");
    cout << "\n*****************************test 12" << endl; 
    // print_vector(ts);
    start = time_point.now();
    fabba = fABBA::ABBA(tol, alpha, "lexi", 1, series_size, true);
    symbols= fabba.fit_transform(ts);
    r_fabba_series = fabba.inverse_transform(ts[0]);
    end = time_point.now();

    // cout << symbols << endl;
    runtime = static_cast<chrono::duration<double>>(end - start);
    error = norm(ts - r_fabba_series);
    //dtw =  DTW(ts, r_fabba_series, dist_euclidean, 1);
    errors.push_back(error);
    //errors_dtw.push_back(dtw);

    runtimes.push_back(runtime.count());
    cout << "runtime: " << runtime.count()<<" seconds" << endl;
    //cout << "error:" << error << endl; 
    test = {"test 12", ts};
    benchmarks.push_back(test); 
    test = {"reconst test 12", r_fabba_series};
    benchmarks.push_back(test); 


    ts = generate_random_sequence(series_size, "normal");
    cout << "\n*****************************test 13" << endl; 
    // print_vector(ts);
    start = time_point.now();
    fabba = fABBA::ABBA(tol, alpha, "lexi", 1, series_size, true);
    symbols= fabba.fit_transform(ts);
    r_fabba_series = fabba.inverse_transform(ts[0]);
    end = time_point.now();

    // cout << symbols << endl;
    runtime = static_cast<chrono::duration<double>>(end - start);
    error = norm(ts - r_fabba_series);
    //dtw =  DTW(ts, r_fabba_series, dist_euclidean, 1);
    errors.push_back(error);
    //errors_dtw.push_back(dtw);

    runtimes.push_back(runtime.count());
    cout << "runtime: " << runtime.count()<<" seconds" << endl;
    //cout << "error:" << error << endl; 
    test = {"test 13", ts};
    benchmarks.push_back(test); 
    test = {"reconst test 13", r_fabba_series};
    benchmarks.push_back(test); 


    ts = generate_random_sequence(series_size, "normal");
    cout << "\n*****************************test 14" << endl; 
    // print_vector(ts);
    start = time_point.now();
    fabba = fABBA::ABBA(tol, alpha, "lexi", 1, series_size, true);
    symbols= fabba.fit_transform(ts);
    r_fabba_series = fabba.inverse_transform(ts[0]);
    end = time_point.now();

    // cout << symbols << endl;
    runtime = static_cast<chrono::duration<double>>(end - start);
    error = norm(ts - r_fabba_series);
    //dtw =  DTW(ts, r_fabba_series, dist_euclidean, 1);
    errors.push_back(error);
    //errors_dtw.push_back(dtw);

    runtimes.push_back(runtime.count());
    cout << "runtime: " << runtime.count()<<" seconds" << endl;
    //cout << "error:" << error << endl; 
    test = {"test 14", ts};
    benchmarks.push_back(test); 
    test = {"reconst test 14", r_fabba_series};
    benchmarks.push_back(test); 


    ts = generate_random_sequence(series_size, "normal");
    cout << "\n*****************************test 15" << endl; 
    // print_vector(ts);
    start = time_point.now();
    fabba = fABBA::ABBA(tol, alpha, "lexi", 1, series_size, true);
    symbols= fabba.fit_transform(ts);
    r_fabba_series = fabba.inverse_transform(ts[0]);
    end = time_point.now();

    // cout << symbols << endl;
    runtime = static_cast<chrono::duration<double>>(end - start);
    error = norm(ts - r_fabba_series);
    //dtw =  DTW(ts, r_fabba_series, dist_euclidean, 1);
    errors.push_back(error);
    //errors_dtw.push_back(dtw);

    runtimes.push_back(runtime.count());
    cout << "runtime: " << runtime.count()<<" seconds" << endl;
    //cout << "error:" << error << endl; 
    test = {"test 15", ts};
    benchmarks.push_back(test); 
    test = {"reconst test 15", r_fabba_series};
    benchmarks.push_back(test); 


    ts = generate_random_sequence(series_size, "normal");
    cout << "\n*****************************test 16" << endl; 
    // print_vector(ts);
    start = time_point.now();
    fabba = fABBA::ABBA(tol, alpha, "lexi", 1, series_size, true);
    symbols= fabba.fit_transform(ts);
    r_fabba_series = fabba.inverse_transform(ts[0]);
    end = time_point.now();

    // cout << symbols << endl;
    runtime = static_cast<chrono::duration<double>>(end - start);
    error = norm(ts - r_fabba_series);
    //dtw =  DTW(ts, r_fabba_series, dist_euclidean, 1);
    errors.push_back(error);
    //errors_dtw.push_back(dtw);

    runtimes.push_back(runtime.count());
    cout << "runtime: " << runtime.count()<<" seconds" << endl;
    //cout << "error:" << error << endl; 
    test = {"test 16", ts};
    benchmarks.push_back(test); 
    test = {"reconst test 16", r_fabba_series};
    benchmarks.push_back(test); 


    ts = generate_random_sequence(series_size, "normal");
    cout << "\n*****************************test 17" << endl; 
    // print_vector(ts);
    start = time_point.now();
    fabba = fABBA::ABBA(tol, alpha, "lexi", 1, series_size, true);
    symbols= fabba.fit_transform(ts);
    r_fabba_series = fabba.inverse_transform(ts[0]);
    end = time_point.now();

    // cout << symbols << endl;
    runtime = static_cast<chrono::duration<double>>(end - start);
    error = norm(ts - r_fabba_series);
    //dtw =  DTW(ts, r_fabba_series, dist_euclidean, 1);
    errors.push_back(error);
    //errors_dtw.push_back(dtw);

    runtimes.push_back(runtime.count());
    cout << "runtime: " << runtime.count()<<" seconds" << endl;
    //cout << "error:" << error << endl; 
    test = {"test 17", ts};
    benchmarks.push_back(test); 
    test = {"reconst test 17", r_fabba_series};
    benchmarks.push_back(test); 


    ts = generate_random_sequence(series_size, "normal");
    cout << "\n*****************************test 18" << endl; 
    // print_vector(ts);
    start = time_point.now();
    fabba = fABBA::ABBA(tol, alpha, "lexi", 1, series_size, true);
    symbols= fabba.fit_transform(ts);
    r_fabba_series = fabba.inverse_transform(ts[0]);
    end = time_point.now();

    // cout << symbols << endl;
    runtime = static_cast<chrono::duration<double>>(end - start);
    error = norm(ts - r_fabba_series);
    //dtw =  DTW(ts, r_fabba_series, dist_euclidean, 1);
    errors.push_back(error);
    //errors_dtw.push_back(dtw);

    runtimes.push_back(runtime.count());
    cout << "runtime: " << runtime.count()<<" seconds" << endl;
    //cout << "error:" << error << endl; 
    test = {"test 18", ts};
    benchmarks.push_back(test); 
    test = {"reconst test 18", r_fabba_series};
    benchmarks.push_back(test); 


    ts = generate_random_sequence(series_size, "normal");
    cout << "\n*****************************test 19" << endl; 
    // print_vector(ts);
    start = time_point.now();
    fabba = fABBA::ABBA(tol, alpha, "lexi", 1, series_size, true);
    symbols= fabba.fit_transform(ts);
    r_fabba_series = fabba.inverse_transform(ts[0]);
    end = time_point.now();

    // cout << symbols << endl;
    runtime = static_cast<chrono::duration<double>>(end - start);
    error = norm(ts - r_fabba_series);
    //dtw =  DTW(ts, r_fabba_series, dist_euclidean, 1);
    errors.push_back(error);
    //errors_dtw.push_back(dtw);

    runtimes.push_back(runtime.count());
    cout << "runtime: " << runtime.count()<<" seconds" << endl;
    //cout << "error:" << error << endl; 
    test = {"test 19", ts};
    benchmarks.push_back(test); 
    test = {"reconst test 19", r_fabba_series};
    benchmarks.push_back(test); 


    ts = generate_random_sequence(series_size, "normal");
    cout << "\n*****************************test 20" << endl; 
    // print_vector(ts);
    start = time_point.now();
    fabba = fABBA::ABBA(tol, alpha, "lexi", 1, series_size, true);
    symbols= fabba.fit_transform(ts);
    r_fabba_series = fabba.inverse_transform(ts[0]);
    end = time_point.now();

    // cout << symbols << endl;
    runtime = static_cast<chrono::duration<double>>(end - start);
    error = norm(ts - r_fabba_series);
    //dtw =  DTW(ts, r_fabba_series, dist_euclidean, 1);
    errors.push_back(error);
    //errors_dtw.push_back(dtw);

    runtimes.push_back(runtime.count());
    cout << "runtime: " << runtime.count()<<" seconds" << endl;
    //cout << "error:" << error << endl; 
    test = {"test 20", ts};
    benchmarks.push_back(test); 
    test = {"reconst test 20", r_fabba_series};
    benchmarks.push_back(test); 


    vector<pair<string, vector<double>>> performance = {{"runtime", runtimes}, {"2-norm", errors}}; //, {"DTW", errors_dtw}}; 
    data_to_csv(performance, "performance.csv", true);
    data_to_csv(benchmarks, "benchmark_test.csv", true);

    return  0;
}

 #ifndef KRYLOV_UTILS
 #define KRYLOV_UTILS


#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <chrono>
#include <map>
using namespace std::chrono;

void krylov_L(int L);

void cout_vector(std::vector<int> const &input);

std::vector<std::vector<int>> get_H_ring(int L, int X);

std::vector<std::vector<int>> get_H_hopp(int L, int X);

std::vector<std::vector<char>> sort_basis(std::vector<std::vector<int>> H_ring,
                                              std::map<int, std::vector<char>> *ring_map,
                                          std::vector<std::vector<int>> H_hopp,
                                          std::map<int, std::vector<char>> *hop_map,
                                          int L) ;

void fbasis(int L, std::vector<std::vector<char>> basis, std::string bname);

void fH(std::map<int, std::vector<char>> *h_map,
        std::string hname);

std::vector<char> get_config(int L);

bool insert_new(std::vector<std::vector<char>> *basis, std::vector<char> psi_f);

bool hopp_cond_flip(std::vector<char> *psi_f, std::vector<int> H_i);

bool ring_cond_flip(std::vector<char> *psi_f, std::vector<int> H_i);
#endif
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

void cout_vector(std::vector<int> const &input) {
  std::copy(input.begin(), input.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
}

std::vector<std::vector<int>> get_H_ring(int L, int X);

std::vector<std::vector<int>> get_H_hopp(int L, int X);

std::vector<std::vector<char>> sort_basis(std::vector<std::vector<int>> H_ring,
                                          std::vector<std::vector<int>> H_hopp, int L);

void fbasis(int L, std::vector<std::vector<char>> basis, std::string bname);

void fH(std::vector<std::vector<char>> basis, std::vector<std::vector<int>> H,
        std::string hname, bool cond_flip(std::vector<char> *, std::vector<int>));

std::vector<char> get_config(int L);

bool insert_new(std::vector<std::vector<char>> *basis, std::vector<char> psi_f);

bool hopp_cond_flip(std::vector<char> *psi_f, std::vector<int> H_i);

bool ring_cond_flip(std::vector<char> *psi_f, std::vector<int> H_i);

void krylov_L(int L);

int main(int argc, char *argv[]) {
  int L;
  
  if (argc >= 2)
  {
    for (int i = 1; i < argc; i++)
    {
      L = std::stoi(argv[i]);
      std::cout << "L[" << i << "] = " <<  L << " (from args)"<< std::endl;
      krylov_L(L);
      std::cout << "========================================= " << std::endl;
    }
  }
  else
  {
    L = 12;
    std::cout << "Def L = " <<  L << std::endl;
    krylov_L(L);
  }


  std::cout << "-- END PROGRAM --:" << std::endl
            << "Have a nive day! \342\230\272" << std::endl;
  return 0;
}

void krylov_L(int L)
{
  int X = 2;
  std::cout << "1. Terms in H_ring: " << std::endl;
  std::vector<std::vector<int>> H_ring = get_H_ring(L, X);

  std::cout << "2. Terms in H_hopp: " << std::endl;
  std::vector<std::vector<int>> H_hopp = get_H_hopp(L, X);

  std::cout << "3. H_ring size = " << H_ring.size() << std::endl;
  std::cout << "4. H_hopp size = " << H_hopp.size() << std::endl;

  std::vector<std::vector<char>> basis = sort_basis(H_ring, H_hopp, L);

  fbasis(L, basis, "basis_L" + std::to_string(L) + ".dat");
  fH(basis, H_ring, "matrix_ring_L" + std::to_string(L) + ".dat", ring_cond_flip);
  fH(basis, H_hopp, "matrix_hopp_L" + std::to_string(L) + ".dat", hopp_cond_flip);
}

std::vector<std::vector<int>> get_H_ring(int L, int X) {
  /*
  Each H_i is a ring term - for dimer bonds in a ring at the i-th ring of the ladder.
  */
  std::vector<std::vector<int>> H_ring;

  for (int i = 0; i < L -1 ; i++) {
    std::vector<int> H_i = {3 * i, 3 * ((i + 1) % L), 3 * i + 1, 3 * i + 2};

    //cout_vector(H_i);
    H_ring.push_back(H_i);
  }

  return H_ring;
}

std::vector<std::vector<int>> get_H_hopp(int L, int X) {
  std::vector<std::vector<int>> H_hopp;
  std::vector<int> H_i;

  for (int i = 0; i < L; i++) {


    // 6.
    H_i = {3 * ((i + 3) % L) + 0, 3 * ((i + 2) % L) + 2, 3 * ((i + 3) % L) + 2,    
           3 * ((i + 2) % L) + 1, 3 * ((i + 2) % L) + 0, 3 * ((i + 1) % L) + 1};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    // 7.
    H_i = {3 * ((i + 3) % L) + 0, 3 * ((i + 2) % L) + 2, 3 * ((i + 3) % L) + 2,
           3 * ((i + 3) % L) + 1, 3 * ((i + 4) % L) + 0, 3 * ((i + 4) % L) + 1};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    
    // 2.
    H_i = {3 * ((i + 2) % L) + 0, 3 * ((i + 1) % L) + 1, 3 * ((i + 2) % L) + 1,
           3 * ((i + 1) % L) + 2, 3 * ((i + 1) % L) + 0, 3 * (i % L) + 2};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    // 3.
    H_i = {3 * ((i + 1) % L) + 0, 3 * (i % L) + 1, 3 * ((i + 1) % L) + 1,
           3 * ((i + 1) % L) + 2, 3 * ((i + 2) % L) + 0, 3 * ((i + 2) % L) + 2};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    
    // 1.
    H_i = {3 * ((i + 1) % L) + 1, 3 * ((i + 2) % L) + 0, 3 * ((i + 2) % L) + 1,
                            3 * ((i + 1) % L) + 0, 3 * (i % L) + 2 ,3 * ((i + 1) % L) + 2};
    
    //Conditions verify no Hamiltonian term allows defect at ring i=0 to hop.
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }


    // 4.
    H_i = {3 * ((i + 1) % L) + 1, 3 * ((i + 1) % L) + 0, 3 * (i % L) + 1,
           3 * ((i + 2) % L) + 0, 3 * ((i + 1) % L) + 2, 3 * ((i + 2) % L) + 2};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

       // 9.
    H_i = {3 * ((i + 2) % L) + 1, 3 * ((i + 3) % L) + 0, 3 * ((i + 3) % L) + 1,
           3 * ((i + 1) % L) + 1, 3 * ((i + 0) % L) + 1, 3 * ((i + 1) % L)};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    // 10.
    H_i = {3 * ((i + 3) % L) + 1, 3 * ((i + 3) % L) + 0, 3 * ((i + 2) % L) + 1,
           3 * ((i + 4) % L) + 1, 3 * ((i + 5) % L) + 0, 3 * ((i + 5) % L) + 1};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    // 5.
    H_i = {3 * ((i + 1) % L) + 2, 3 * ((i + 2) % L) + 0, 3 * ((i + 2) % L) + 2,
           3 * ((i + 1) % L) + 0, 3 * (i % L) + 1, 3 * ((i + 1) % L) + 1};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    // 8.
    H_i = {3 * ((i + 3) % L) + 2, 3 * ((i + 3) % L) + 0, 3 * ((i + 2) % L) + 2, 
           3 * ((i + 4) % L) + 0, 3 * ((i + 3) % L) + 1, 3 * ((i + 4) % L) + 1};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }


    // 11.
    H_i = {3 * ((i + 2) % L) + 2, 3 * ((i + 3) % L) + 0, 3 * ((i + 3) % L) + 2, 
           3 * ((i + 1) % L) + 2, 3 * ((i + 0) % L) + 2, 3 * ((i + 1) % L)};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    // 12.
    H_i = {3 * ((i + 1) % L) + 2, 3 * ((i + 1) % L) + 0, 3 * (i % L) + 2,
           3 * ((i + 2) % L) + 2, 3 * ((i + 3) % L) + 0, 3 * ((i + 3) % L) + 2};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }
  } // End for (int i = 0; i < L; i++)

  return H_hopp;
}

std::vector<std::vector<char>> sort_basis(std::vector<std::vector<int>> H_ring,
                                          std::vector<std::vector<int>> H_hopp, int L) {
  std::vector<std::vector<char>> basis{get_config(L)};

  bool found = 1;
  while (found) {
    found = 0;
    //int n = basis.size();
    for (int k = 0; k < basis.size(); k++) {
      for (auto H_i : H_ring) {
        std::vector<char> psi_f(basis[k]);
        if (ring_cond_flip(&psi_f, H_i)) {
          // Adds H_i|psi_f> to basis if it conserves |psi_f>'s defect number.
          found = insert_new(&basis, psi_f) || found;
        }
      }

      for (auto H_i : H_hopp) {
        std::vector<char> psi_g(basis[k]);
        if (hopp_cond_flip(&psi_g, H_i)) {
          found = insert_new(&basis, psi_g) || found;
        }
      }
    }// End for k
  }
  std::cout << "end sort_basis " << basis.size()<< std::endl;
  return basis;
}

void fH(std::vector<std::vector<char>> basis, std::vector<std::vector<int>> H,
        std::string hname, bool cond_flip(std::vector<char> *, std::vector<int>)) {
  std::fstream dateh(hname, std::ios::out | std::ios::binary);
  int n_configs = basis.size();
  int t = 1; //Transition amplitude.
  dateh.write((char *)&n_configs, sizeof(int));
  dateh.write((char *)&n_configs, sizeof(int));
  long int n_nonzeros = 0;
  
  for (int k = 0; k < n_configs; k++) {
    for (auto H_i : H) {
      std::vector<char> psi_f(basis[k]);
      if (cond_flip(&psi_f, H_i)) {
        int index_f = std::distance(basis.begin(), std::find(basis.begin(), basis.end(), psi_f));
        if (index_f < n_configs) {// H_i[k, index_f] = 1. So <basis[index_f]|H_i|basis[k]> = t
          dateh.write((char *)&k, sizeof(int));
          dateh.write((char *)&index_f, sizeof(int));
          dateh.write((char *)&t, sizeof(int));
          n_nonzeros++;
        } else
          std::cerr << "Problem! " << hname << std::endl;
      }
    }
  }
  std::cerr << "Nonzeros in " << hname << " : " << n_nonzeros << std::endl;
  dateh.seekp(4);
  dateh.write((char *)&n_nonzeros, sizeof(int));
  dateh.close();
}

bool hopp_cond_flip(std::vector<char> *psi_f, std::vector<int> H_i) {
  //If a defect is present at edge 0 it hops to edge 3.
  if (((*psi_f)[H_i[0]] == 1) && ((*psi_f)[H_i[3]] == 0) &&
      ((*psi_f)[H_i[1]] + (*psi_f)[H_i[2]] == 1) &&
      ((*psi_f)[H_i[4]] + (*psi_f)[H_i[5]] == 1)) {

    (*psi_f)[H_i[0]] = 0;//1 - (*psi_f)[H_i[0]];
    (*psi_f)[H_i[3]] = 1;//1 - (*psi_f)[H_i[3]];
    return true;
  }
  
  return false;
}

bool ring_cond_flip(std::vector<char> *psi_f, std::vector<int> H_i) {
  // If the i-th ring of psi_f is = flip to || and vice versa (states with out defects). Otherwise do notihng.
  bool i_ring_II = ((*psi_f)[H_i[0]] == 1) && ((*psi_f)[H_i[1]] == 1) &&
                   ((*psi_f)[H_i[2]] == 0) && ((*psi_f)[H_i[3]] == 0) ;
  bool i_ring_Z =  ((*psi_f)[H_i[0]] == 0) && ((*psi_f)[H_i[1]] == 0) &&
                   ((*psi_f)[H_i[2]] == 1) && ((*psi_f)[H_i[3]] == 1) ;
  
  if ( i_ring_II || i_ring_Z ) {
    (*psi_f)[H_i[0]] = 1 - (*psi_f)[H_i[0]];
    (*psi_f)[H_i[1]] = 1 - (*psi_f)[H_i[1]];
    (*psi_f)[H_i[2]] = 1 - (*psi_f)[H_i[2]];
    (*psi_f)[H_i[3]] = 1 - (*psi_f)[H_i[3]];
  }
  return i_ring_II || i_ring_Z;
}

void fbasis(int L, std::vector<std::vector<char>> basis, std::string bname) {
  std::fstream dateib(bname, std::ios::out | std::ios::binary);

  int n_configs = basis.size();
  dateib.write((char *)&n_configs, sizeof(int));
  dateib.write((char *)&(L), sizeof(int));

  for (auto it : basis) {
    std::copy(it.begin(), it.end(), std::ostream_iterator<char>(dateib, ""));
  }
  dateib.close();
  std::cout << "Dimension: " << n_configs << std::endl;
}

bool insert_new(std::vector<std::vector<char>> *basis, std::vector<char> psi_f) {
  bool res = not binary_search(basis->begin(), basis->end(), psi_f);
  if (res) {
    basis->push_back(psi_f);
    sort(basis->begin(), basis->end());
  }
  return res;
}

std::vector<char> get_config(int L) {
  //Creates a state with defects at ring 0 and ring L/2.
  std::vector<char> config(3 * L, 0);
  config[0] = 1;
  config[2] = 1;

  for (int i = 1; i < L / 2 + 1; i++)
    config[3 * i + 1 + (i + 1) % 2] = 1;
  
  for (int i = L / 2 + 1; i < L; i++)
    config[3 * i] = 1;
    
  return config;
}

/*
std::vector<std::vector<int>> get_H_hopp_old(int L, int X) {
  //
  std::vector<std::vector<int>> H_hopp;

  for (int i = 0; i < L; i++) {
    
    // 1.
    std::vector<int>  H_i = {3 * ((i + 2) % L) + 0, 3 * ((i + 1) % L) + 2, 3 * ((i + 2) % L) + 2,    
                             3 * ((i + 1) % L) + 1, 3 * ((i + 1) % L) + 0, 3 * (i % L) + 1};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    // 2.
    H_i = {3 * ((i + 3) % L) + 0, 3 * ((i + 2) % L) + 2, 3 * ((i + 3) % L) + 2,
           3 * ((i + 3) % L) + 1, 3 * ((i + 4) % L) + 0, 3 * ((i + 4) % L) + 1};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    // 3.
    H_i = {3 * ((i + 3) % L) + 0, 3 * ((i + 2) % L) + 1, 3 * ((i + 3) % L) + 1,
           3 * ((i + 2) % L) + 2, 3 * ((i + 2) % L) + 0, 3 * ((i + 1) % L) + 2};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    // 4.
    H_i = {3 * ((i + 3) % L) + 0, 3 * ((i + 2) % L) + 1, 3 * ((i + 3) % L) + 1,
           3 * ((i + 3) % L) + 2, 3 * ((i + 4) % L) + 0, 3 * ((i + 4) % L) + 2};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      cout_vector(H_i);
      H_hopp.push_back(H_i);
    }
    
    // 5.
    H_i = {3 * ((i + 2) % L) + 1, 3 * ((i + 3) % L) + 0, 3 * ((i + 3) % L) + 1,
           3 * ((i + 2) % L) + 0, 3 * ((i + 1) % L) + 2 ,3 * ((i + 2) % L) + 2};
    
    //Conditions verify no Hamiltonian term allows defect at ring i=0 to hop.
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    // 6.
    H_i = {3 * ((i + 3) % L) + 1, 3 * ((i + 3) % L) + 0, 3 * ((i + 2) % L) + 1,
           3 * ((i + 4) % L) + 0, 3 * ((i + 3) % L) + 2, 3 * ((i + 4) % L) + 2};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      cout_vector(H_i);
      H_hopp.push_back(H_i);
    }
  

    // 7.
    H_i = {3 * ((i + 2) % L) + 1, 3 * ((i + 3) % L) + 0, 3 * ((i + 3) % L) + 1,
           3 * ((i + 1) % L) + 1, 3 * ((i + 0) % L) + 1, 3 * ((i + 1) % L)};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    // 8.
    H_i = {3 * ((i + 3) % L) + 1, 3 * ((i + 3) % L) + 0, 3 * ((i + 2) % L) + 1,
           3 * ((i + 4) % L) + 1, 3 * ((i + 5) % L) + 0, 3 * ((i + 5) % L) + 1};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    // 9.
    H_i = {3 * ((i + 2) % L) + 2, 3 * ((i + 3) % L) + 0, 3 * ((i + 3) % L) + 2,
           3 * ((i + 2) % L) + 0, 3 * ((i + 1) % L) + 1, 3 * ((i + 2) % L) + 1};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      cout_vector(H_i);
      H_hopp.push_back(H_i);
    }
  
    // 10.
    H_i = {3 * ((i + 3) % L) + 2, 3 * ((i + 3) % L) + 0, 3 * ((i + 2) % L) + 2, 
           3 * ((i + 4) % L) + 0, 3 * ((i + 3) % L) + 1, 3 * ((i + 4) % L) + 1};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    // 11.
    H_i = {3 * ((i + 2) % L) + 2, 3 * ((i + 3) % L) + 0, 3 * ((i + 3) % L) + 2, 
           3 * ((i + 1) % L) + 2, 3 * ((i + 0) % L) + 2, 3 * ((i + 1) % L)};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    // 12.
    H_i = {3 * ((i + 3) % L) + 2, 3 * ((i + 3) % L) + 0, 3 * ((i + 2) % L) + 2,
           3 * ((i + 4) % L) + 2, 3 * ((i + 5) % L) + 0, 3 * ((i + 5) % L) + 2};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      cout_vector(H_i);
      H_hopp.push_back(H_i);
    }
  } // End for (int i = 0; i < L; i++)

  return H_hopp;
}
*/
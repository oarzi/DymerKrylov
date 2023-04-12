#include "krylov_points_utils.h"
#include <deque>
#include <cmath>

void krylov_L(int L) {
  auto start = high_resolution_clock::now();
  int X = 1;
  std::cout << "Terms in H_ring: " << std::endl;
  std::vector<std::vector<int>> H_ring = get_H_ring(L);
  std::cout << "H_ring size = " << H_ring.size() << std::endl;

  std::cout << "Terms in H_hopp: " << std::endl;
  std::vector<std::vector<int>> H_hopp = get_H_hopp(L, X);
  std::cout << "H_hopp size = " << H_hopp.size() << std::endl;

  std::map<int, std::vector<int>*> ring_map;
  std::map<int, std::vector<int>*> hop_map;
  std::map<int, std::vector<char>> basis = sort_basis(H_ring, &ring_map, H_hopp, &hop_map, L);

  fbasis(L, basis, "basis_L" + std::to_string(L) + ".dat");
  fH(&ring_map, "matrix_ring_L" + std::to_string(L) + ".dat");
  fH(&hop_map, "matrix_hopp_L" + std::to_string(L) + ".dat");

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  std::cout << "Duration:" << duration.count() << std::endl;
}

void cout_vector(std::vector<int> const &input) {
  std::copy(input.begin(), input.end(),
            std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
}

std::vector<std::vector<int>> get_H_ring(int L) {
  /*
  Each H_i is a ring term - for dimer bonds in a ring at the i-th ring of the
  ladder.
  */
  std::vector<std::vector<int>> H_ring;

  for (int i = 1; i < L - 1; i++) {
    std::vector<int> H_i = {3 * i, 3 * ((i + 1) % L), 3 * i + 1, 3 * i + 2};

    // cout_vector(H_i);
    H_ring.push_back(H_i);
  }

  return H_ring;
}

std::vector<std::vector<int>> get_H_hopp(int L, int X) {
  //
  std::vector<std::vector<int>> H_hopp;
  std::vector<int> H_i;

  for (int i = 1; i < L - 1; i++) {
    
    // Top-right
    H_i = {3 * i + 2, 3 * ((i + 1) % L) + 2, 3 * ((i + 1) % L), 3 * i + 1};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) ) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }
    
    // Top-left
    H_i = {3 * (i - 1) + 2, 3 * i  + 0, 3 * i + 2, 3 * ((i + 1)%L)};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X)) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    // Bottom-right
    H_i = {3 * i + 1, 3 * ((i + 1) % L) + 1, 3 * ((i + 1)%L), 3 * i +2};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X)) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    // Bottom-leftuy8
    H_i = {3 * (i - 1) + 1, 3 * i, 3 * i + 1, 3 * ((i + 1)%L)};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X)) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }


  } // End for (int i = 0; i < L; i++)

  return H_hopp;
}

std::map<int, std::vector<char>> sort_basis(std::vector<std::vector<int>> H_ring,
                                          std::map<int, std::vector<int>*> *ring_map,
                                          std::vector<std::vector<int>> H_hopp,
                                          std::map<int, std::vector<int>*> *hop_map,
                                          int L) {
  std::map<int, std::vector<char>> basis;
  basis[0] = get_config(L);

  std::map<std::vector<char>, int> names; 
  names[basis[0]] = 0;
  int count = 1;
  std::deque<std::vector<char>> states{basis[0]};

  while (!states.empty()) {
    std::vector<char> curr = states.front();
    int name = names[curr];
    
    (*ring_map)[name] = new std::vector<int>();
    (*hop_map)[name] = new std::vector<int>();
    states.pop_front();

    for (auto H_i : H_ring) {
      std::vector<char> psi_f(curr);
      if (ring_cond_flip(&psi_f, H_i)) {
        if (auto search = names.find(psi_f); search == names.end()){
            names[psi_f] = count;
            basis[count] = psi_f; 
            count++;
            states.push_back(psi_f);
            //std::cout << count << std::endl;
          }

        (*ring_map)[name]->push_back(names[psi_f]);
      }
    }

    for (auto H_i : H_hopp) {
      std::vector<char> psi_g(curr);
      if (hopp_cond_flip(&psi_g, H_i)) {
        if (auto search = names.find(psi_g); search == names.end()){
            names[psi_g] = count;
            basis[count] = psi_g; 
            count++;
            states.push_back(psi_g);
            //std::cout << count << std::endl;
          }
        (*hop_map)[name]->push_back(names[psi_g]);
      }
    }
  }

 
  std::cout << "end sort_basis " << basis.size() << std::endl;
  return basis;
}

void fH(std::map<int, std::vector<int>*> *h_map,
        std::string hname){
  std::fstream dateh(hname, std::ios::out | std::ios::binary);
  int n_configs = h_map->size();
  int t = 1; // Transition amplitude.
  dateh.write((char *)&n_configs, sizeof(int));
  dateh.write((char *)&n_configs, sizeof(int));
  long int n_nonzeros = 0;

  for (auto const& pair : *h_map) {
    int key = pair.first;
    std::vector<int> to_write = *pair.second;
    //std::cout << key << ": ";
    for (int name : to_write)
      {
          //std::cout << name << ", ";
          int _name = name;
          dateh.write((char *)&key, sizeof(int));
          dateh.write((char *)&(_name), sizeof(int));
          dateh.write((char *)&t, sizeof(int));
          n_nonzeros++;
      }
    //std::cout << std::endl;
      
  }

  std::cout << "Nonzeros in " << hname << " : " << n_nonzeros << std::endl;
  dateh.seekp(4);
  dateh.write((char *)&n_nonzeros, sizeof(int));
  dateh.close();
}


bool hopp_cond_flip(std::vector<char> *psi_f, std::vector<int> H_i) {
  // If a defect is present at edge 0 it hops to edge 3.
  bool hop_cond = ((*psi_f)[H_i[0]] == (*psi_f)[H_i[1]]) &&
                  ((*psi_f)[H_i[1]] == (*psi_f)[H_i[2]]) && ((*psi_f)[H_i[0]] == 0) && ((*psi_f)[H_i[3]] == 1);
  
  if (hop_cond) {
    int temp = (*psi_f)[H_i[2]]; 
    (*psi_f)[H_i[2]] = (*psi_f)[H_i[3]];
    (*psi_f)[H_i[3]] = temp;
  }

  return hop_cond;
}

bool ring_cond_flip(std::vector<char> *psi_f, std::vector<int> H_i) {
  // If the i-th ring of psi_f is = flip to || and vice versa (states with out
  // defects). Otherwise do notihng.

  bool i_ring = ((*psi_f)[H_i[0]] == (*psi_f)[H_i[1]]) &&
                ((*psi_f)[H_i[2]] == (*psi_f)[H_i[3]]) && (((*psi_f)[H_i[0]] != (*psi_f)[H_i[2]] ));

  if (i_ring) {
    (*psi_f)[H_i[0]] = 1 - (*psi_f)[H_i[0]];
    (*psi_f)[H_i[1]] = 1 - (*psi_f)[H_i[1]];
    (*psi_f)[H_i[2]] = 1 - (*psi_f)[H_i[2]];
    (*psi_f)[H_i[3]] = 1 - (*psi_f)[H_i[3]];
  }
  return i_ring;
}

void fbasis(int L, std::map<int, std::vector<char>> basis, std::string bname) {
  std::fstream dateib(bname, std::ios::out | std::ios::binary);

  int n_configs = basis.size();
  dateib.write((char *)&n_configs, sizeof(int));
  dateib.write((char *)&(L), sizeof(int));
  
  for (int i=0 ;i < basis.size(); i++) {
    std::copy(basis[i].begin(), basis[i].end(), std::ostream_iterator<char>(dateib, ""));
  }
  dateib.close();
  std::cout << "Dimension: " << n_configs << std::endl;
}

bool insert_new(std::vector<std::vector<char>> *basis,
                std::vector<char> psi_f) {
  bool res = not binary_search(basis->begin(), basis->end(), psi_f);
  if (res) {
    basis->insert(std::lower_bound(basis->begin(), basis->end(), psi_f), psi_f);
  }

  return res;
}

std::vector<char> get_config(int L) {
  // Creates a state with defects at ring 0 and ring L/2.
  std::vector<char> config(3 * L, 0);
  config[0] = 0;
  config[2] = 0;
  int defect = std::floor(L/2);
  
  for (int i = 0; i < defect; i++)
    config[3 * i + 1 + i % 2] = 1;

  for (int i = defect + 1; i < L; i=i+2)
    {
        config[3 * i + 1] = 1;
        config[3 * i + 2] = 1;
    }
 
 config[3*L - 1] = 0;
 config[3*L - 2] = 0;
 if ((L - defect) %2 == 0)
 {
    config[3*L - 3] = 1;
 }

  return config;
}
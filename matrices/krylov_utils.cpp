#include "krylov_utils.h"
#include <deque>


void krylov_L(int L) {
  auto start = high_resolution_clock::now();
  int X = 2;
  std::cout << "Terms in H_ring: " << std::endl;
  std::vector<std::vector<int>> H_ring = get_H_ring(L, X);
  std::cout << "H_ring size = " << H_ring.size() << std::endl;

  std::cout << "Terms in H_hopp: " << std::endl;
  std::vector<std::vector<int>> H_hopp = get_H_hopp(L, X);
  std::cout << "H_hopp size = " << H_hopp.size() << std::endl;

  std::map<int, std::vector<char>> ring_map;
  std::map<int, std::vector<char>> hop_map;
  std::vector<std::vector<char>> basis = sort_basis(H_ring, &ring_map, H_hopp, &hop_map, L);

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

std::vector<std::vector<int>> get_H_ring(int L, int X) {
  /*
  Each H_i is a ring term - for dimer bonds in a ring at the i-th ring of the
  ladder.
  */
  std::vector<std::vector<int>> H_ring;

  for (int i = 0; i < L - 1; i++) {
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

  for (int i = 0; i < L; i++) {

    H_i = {3 * ((i + 3) % L) + 0, 3 * ((i + 2) % L) + 2, 3 * ((i + 3) % L) + 2,
           3 * ((i + 2) % L) + 1, 3 * ((i + 2) % L) + 0, 3 * ((i + 1) % L) + 1};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    H_i = {3 * ((i + 3) % L) + 0, 3 * ((i + 2) % L) + 2, 3 * ((i + 3) % L) + 2,
           3 * ((i + 3) % L) + 1, 3 * ((i + 4) % L) + 0, 3 * ((i + 4) % L) + 1};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    H_i = {3 * ((i + 2) % L) + 0, 3 * ((i + 1) % L) + 1, 3 * ((i + 2) % L) + 1,
           3 * ((i + 1) % L) + 2, 3 * ((i + 1) % L) + 0, 3 * (i % L) + 2};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    H_i = {3 * ((i + 1) % L) + 0, 3 * (i % L) + 1,       3 * ((i + 1) % L) + 1,
           3 * ((i + 1) % L) + 2, 3 * ((i + 2) % L) + 0, 3 * ((i + 2) % L) + 2};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    H_i = {3 * ((i + 2) % L) + 1, 3 * ((i + 3) % L) + 0, 3 * ((i + 3) % L) + 1,
           3 * ((i + 1) % L) + 1, 3 * ((i + 0) % L) + 1, 3 * ((i + 1) % L)};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

    H_i = {3 * ((i + 2) % L) + 2, 3 * ((i + 3) % L) + 0, 3 * ((i + 3) % L) + 2,
           3 * ((i + 1) % L) + 2, 3 * ((i + 0) % L) + 2, 3 * ((i + 1) % L)};
    if ((H_i[0] > X) && (H_i[1] > X) && (H_i[2] > X) & (H_i[3] > X) &&
        (H_i[4] > X) && (H_i[5] > X)) {
      // cout_vector(H_i);
      H_hopp.push_back(H_i);
    }

  } // End for (int i = 0; i < L; i++)

  return H_hopp;
}

std::vector<std::vector<int>> get_H_hopp_old(int L, int X) {
  //
  std::vector<std::vector<int>> H_hopp;

  for (int i = 0; i < L; i++) {

    // 1.
    std::vector<int> H_i = {3 * ((i + 2) % L) + 0, 3 * ((i + 1) % L) + 2,
                            3 * ((i + 2) % L) + 2, 3 * ((i + 1) % L) + 1,
                            3 * ((i + 1) % L) + 0, 3 * (i % L) + 1};
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
           3 * ((i + 2) % L) + 0, 3 * ((i + 1) % L) + 2, 3 * ((i + 2) % L) + 2};

    // Conditions verify no Hamiltonian term allows defect at ring i=0 to hop.
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

// std::vector<std::vector<char>> sort_basis(std::vector<std::vector<int>> H_ring,
//                                           std::vector<std::vector<int>> H_hopp,
//                                           int L) {
//   std::vector<std::vector<char>> basis{get_config(L)};


//   std::deque<std::vector<char>> states{basis.front()};

//   while (!states.empty()) {
//     std::vector<char> curr = states.front();
//     states.pop_front();

//     for (auto H_i : H_ring) {
//       std::vector<char> psi_f(curr);
//       if (ring_cond_flip(&psi_f, H_i)) {
//         if (insert_new(&basis, psi_f)) {
//           states.push_back(psi_f);
//         }
//       }
//     }

//     for (auto H_i : H_hopp) {
//       std::vector<char> psi_g(curr);
//       if (hopp_cond_flip(&psi_g, H_i)) {
//         if (insert_new(&basis, psi_g)) {
//           states.push_back(psi_g);
//         }
//       }
//     }
//   }

//   // bool found = 1;
//   // while (found) {
//   //   found = 0;
//   //   // int n = basis.size();
//   //   for (int k = 0; k < basis.size(); k++) {
//   //     for (auto H_i : H_ring) {
//   //       std::vector<char> psi_f(basis[k]);
//   //       if (ring_cond_flip(&psi_f, H_i)) {
//   //         // Adds H_i|psi_f> to basis if it conserves |psi_f>'s defect
//   //         number. found = insert_new(&basis, psi_f) || found;
//   //       }
//   //     }

//   //     for (auto H_i : H_hopp) {
//   //       std::vector<char> psi_g(basis[k]);
//   //       if (hopp_cond_flip(&psi_g, H_i)) {
//   //         found = insert_new(&basis, psi_g) || found;
//   //       }
//   //     }
//   //   } // End for k
//   // }
//   std::cout << "end sort_basis " << basis.size() << std::endl;
//   return basis;
// }

std::vector<std::vector<char>> sort_basis(std::vector<std::vector<int>> H_ring,
                                              std::map<int, std::vector<char>> *ring_map,
                                          std::vector<std::vector<int>> H_hopp,
                                          std::map<int, std::vector<char>> *hop_map,
                                          int L) {
  std::vector<std::vector<char>> basis{get_config(L)};

  std::map<std::vector<char>, int> names; 
  names[basis.front()] = 0;
  int count = 0;
  std::deque<std::vector<char>> states{basis.front()};

  while (!states.empty()) {
    std::vector<char> curr = states.front();
    int name = names[curr];
    (*ring_map)[name];
    (*hop_map)[name];
    states.pop_front();

    for (auto H_i : H_ring) {
      std::vector<char> psi_f(curr);
      if (ring_cond_flip(&psi_f, H_i)) {
        if (insert_new(&basis, psi_f)) {
          states.push_back(psi_f);
          names[psi_f] = count;
          count++;
        }
        (*ring_map)[name].push_back(names[psi_f]);
      }
    }

    for (auto H_i : H_hopp) {
      std::vector<char> psi_g(curr);
      if (hopp_cond_flip(&psi_g, H_i)) {
        if (insert_new(&basis, psi_g)) {
          states.push_back(psi_g);
          names[psi_g] = count;
          count++;
        }
        (*hop_map)[name].push_back(names[psi_g]);
      }
    }
  }

 
  std::cout << "end sort_basis " << basis.size() << std::endl;
  return basis;
}

void fH(std::map<int, std::vector<char>> *h_map,
        std::string hname) {
  std::fstream dateh(hname, std::ios::out | std::ios::binary);
  int n_configs = h_map->size();
  int t = 1; // Transition amplitude.
  dateh.write((char *)&n_configs, sizeof(int));
  dateh.write((char *)&n_configs, sizeof(int));
  long int n_nonzeros = 0;

  for (auto const& pair : *h_map) {
    int key = pair.first;
    for (char name : pair.second)
      {
          dateh.write((char *)&key, sizeof(int));
          dateh.write((char *)&name, sizeof(int));
          dateh.write((char *)&t, sizeof(int));
          n_nonzeros++;
      }
      
  }

  std::cerr << "Nonzeros in " << hname << " : " << n_nonzeros << std::endl;
  dateh.seekp(4);
  dateh.write((char *)&n_nonzeros, sizeof(int));
  dateh.close();
}

// void fH(std::vector<std::vector<char>> basis, std::vector<std::vector<int>> H,
//         std::string hname,
//         bool cond_flip(std::vector<char> *, std::vector<int>)) {
//   std::fstream dateh(hname, std::ios::out | std::ios::binary);
//   int n_configs = basis.size();
//   int t = 1; // Transition amplitude.
//   dateh.write((char *)&n_configs, sizeof(int));
//   dateh.write((char *)&n_configs, sizeof(int));
//   long int n_nonzeros = 0;

//   for (int k = 0; k < n_configs; k++) {
//     for (auto H_i : H) {
//       std::vector<char> psi_f(basis[k]);
//       if (cond_flip(&psi_f, H_i)) {
//         int index_f = std::distance(
//             basis.begin(), std::find(basis.begin(), basis.end(), psi_f));
//         if (index_f < n_configs) { // H_i[k, index_f] = 1. So
//                                    // <basis[index_f]|H_i|basis[k]> = t
//           dateh.write((char *)&k, sizeof(int));
//           dateh.write((char *)&index_f, sizeof(int));
//           dateh.write((char *)&t, sizeof(int));
//           n_nonzeros++;
//         } else
//           std::cerr << "Problem! " << hname << std::endl;
//       }
//     }
//   }
//   std::cerr << "Nonzeros in " << hname << " : " << n_nonzeros << std::endl;
//   dateh.seekp(4);
//   dateh.write((char *)&n_nonzeros, sizeof(int));
//   dateh.close();
// }

bool hopp_cond_flip(std::vector<char> *psi_f, std::vector<int> H_i) {
  // If a defect is present at edge 0 it hops to edge 3.
  bool hop_cond = ((*psi_f)[H_i[0]] != (*psi_f)[H_i[3]]) &&
      ((*psi_f)[H_i[1]] + (*psi_f)[H_i[2]] == 1) &&
      ((*psi_f)[H_i[4]] + (*psi_f)[H_i[5]] == 1);
  
  if (hop_cond) {
    (*psi_f)[H_i[0]] = 1 - (*psi_f)[H_i[0]];
    (*psi_f)[H_i[3]] = 1 - (*psi_f)[H_i[3]];
  }

  return hop_cond;
}

bool ring_cond_flip(std::vector<char> *psi_f, std::vector<int> H_i) {
  // If the i-th ring of psi_f is = flip to || and vice versa (states with out
  // defects). Otherwise do notihng.

  bool i_ring = ((*psi_f)[H_i[0]] == (*psi_f)[H_i[1]] ) &&
                   ((*psi_f)[H_i[2]] ==  (*psi_f)[H_i[3]]) && (((*psi_f)[H_i[0]] != (*psi_f)[H_i[2]] ));

  if (i_ring) {
    (*psi_f)[H_i[0]] = 1 - (*psi_f)[H_i[0]];
    (*psi_f)[H_i[1]] = 1 - (*psi_f)[H_i[1]];
    (*psi_f)[H_i[2]] = 1 - (*psi_f)[H_i[2]];
    (*psi_f)[H_i[3]] = 1 - (*psi_f)[H_i[3]];
  }
  return i_ring;
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
  config[0] = 1;
  config[2] = 1;

  for (int i = 1; i < L / 2 + 1; i++)
    config[3 * i + 1 + (i + 1) % 2] = 1;

  for (int i = L / 2 + 1; i < L; i++)
    config[3 * i] = 1;

  return config;
}
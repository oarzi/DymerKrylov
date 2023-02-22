#include "krylov_points_utils.h"
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char *argv[]) {
  int L;
  std::cout << " --= Krylov vectors for point defects =--" << std::endl;
  if (argc >= 2) {
    for (int i = 1; i < argc; i++) {
      L = std::stoi(argv[i]);
      std::cout << "L = " << L << " (from args)" << std::endl;  
      krylov_L(L);
      std::cout << "========================================= " << std::endl;
    }
  } else {
    L = 11;
    std::cout << "L = " << L << std::endl;
    krylov_L(L);
  }

  std::cout << "-- END PROGRAM --:" << std::endl
            << "Have a nive day! \342\230\272" << std::endl;
  return 0;
}

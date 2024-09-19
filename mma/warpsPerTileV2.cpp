#include <vector>
#include <iostream>

using namespace std;

vector<unsigned> function(int rank, int numWarps, vector<int> shape) {
  vector<unsigned> ret(rank, 1);
  vector<int64_t> shapePerWarp(rank, 1);
  shapePerWarp[rank - 1] = 8;
  shapePerWarp[rank - 2] = 16;
  do {
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (shape[0] / shapePerWarp[0] / ret[0] >=
        shape[1] / (shapePerWarp[1] * 2) / ret[1]) {
      if (ret[0] < shape[0] / shapePerWarp[0]) {
        ret[0] *= 2;
      } else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
    cout << "ret[0]:" << ret[0] << " ret[1]:" << ret[1] << "\n";
  } while (true);
  return ret;
}

int main(int argc, char* argv[]) {
  // 1 4
  for (auto shape : function(2, 4, {16, 16})) {
    cout << shape << " ";
  }
  cout << endl;
  // 2 2
  for (auto shape : function(2, 4, {32, 16})) {
    cout << shape << " ";
  }
  cout << endl;  
  // 4 1
  for (auto shape : function(2, 4, {64, 16})) {
    cout << shape << " ";
  }
  cout << endl;
}

// Test protocol only. Algorithms come from the unmodified pinned OctoMap sources.
#include <octomap/OcTree.h>
#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static octomap::point3d point() {
  float x,y,z; std::cin >> x >> y >> z;
  return octomap::point3d(x,y,z);
}
int main() {
  double res,hit,miss,threshold,low,high;
  std::cin >> res >> hit >> miss >> threshold >> low >> high;
  octomap::OcTree tree(res);
  tree.setProbHit(hit); tree.setProbMiss(miss); tree.setOccupancyThres(threshold);
  tree.setClampingThresMin(low); tree.setClampingThresMax(high);
  std::cout << std::setprecision(17);
  std::string op;
  while(std::cin >> op) {
    if(op == "R") {
      auto origin=point(), end=point(); octomap::KeyRay ray;
      if(!tree.computeRayKeys(origin,end,ray)) {std::cout << "null\n"; continue;}
      std::cout << "["; bool first=true;
      for(auto k:ray) {if(!first) std::cout << ","; first=false;
        std::cout << "[" << k[0] << "," << k[1] << "," << k[2] << "]";}
      std::cout << "]\n";
    } else if(op == "S") {
      int n,lazy,discrete; double maxrange;
      std::cin >> n >> maxrange >> lazy >> discrete;
      auto origin=point(); octomap::Pointcloud scan;
      for(int i=0;i<n;i++) scan.push_back(point());
      tree.insertPointCloud(scan,origin,maxrange,lazy,discrete);
    } else if(op == "U") {
      unsigned x,y,z; float delta; bool lazy;
      std::cin >> x >> y >> z >> delta >> lazy;
      tree.updateNode(octomap::OcTreeKey(x,y,z),delta,lazy);
    } else if(op == "Q") {
      unsigned x,y,z,depth; std::cin >> x >> y >> z >> depth;
      auto n=tree.search(octomap::OcTreeKey(x,y,z),depth);
      if(!n) std::cout << "null\n";
      else std::cout << "[" << n->getLogOdds() << "," << tree.isNodeOccupied(n) << "]\n";
    } else if(op == "B") {
      auto minimum=point(), maximum=point(); bool enabled; std::cin >> enabled;
      tree.setBBXMin(minimum); tree.setBBXMax(maximum); tree.useBBXLimit(enabled);
    } else if(op == "I") tree.updateInnerOccupancy();
    else if(op == "P") tree.prune();
    else if(op == "C") tree.clear();
    else if(op == "D") {
      std::vector<std::array<double,5>> records;
      for(auto it=tree.begin_leafs();it!=tree.end_leafs();++it) {
        auto key=it.getKey(); unsigned span=1u << (16-it.getDepth());
        records.push_back({double(key[0] & ~(span-1)),double(key[1] & ~(span-1)),
                           double(key[2] & ~(span-1)),double(span),it->getLogOdds()});
      }
      std::sort(records.begin(),records.end());
      std::cout << "{\"size\":" << tree.size() << ",\"root\":";
      if(tree.getRoot()) std::cout << tree.getRoot()->getLogOdds(); else std::cout << "null";
      std::cout << ",\"leaves\":[";
      bool first=true;
      for(auto row:records) {if(!first) std::cout << ","; first=false;
        std::cout << "[" << row[0] << "," << row[1] << "," << row[2] << "," << row[3] << "," << row[4] << "]";}
      std::cout << "]}\n";
    } else return 2;
  }
}

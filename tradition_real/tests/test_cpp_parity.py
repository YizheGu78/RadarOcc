"""Compile the vendored upstream methods, replacing only ROS plumbing.

This is an independent C++ oracle, not a rewrite of the Python algorithm.
Requires g++; no ROS/CUDA installation is necessary for this CPU subset.
"""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np
from tradition_real.autoware.costmap import OccupancyGridMap, apply_bbf

ROOT = Path(__file__).resolve().parents[1]


def function(text, signature):
    start = text.index(signature)
    opening = text.index('{', start)
    depth = 1
    cursor = opening + 1
    while depth:
        depth += (text[cursor] == '{') - (text[cursor] == '}')
        cursor += 1
    return text[start:cursor]


def cpp_source():
    source = (ROOT / 'vendor/autoware/occupancy_grid_map.cpp').read_text()
    nav2 = (ROOT / 'vendor/nav2_costmap_2d.hpp').read_text()
    bayes = (ROOT / 'vendor/autoware/binary_bayes_filter_updater.cpp').read_text()
    preamble = r'''
#include <algorithm>
#include <cmath>
#include <climits>
#include <iostream>
#include <string>
#include <vector>
#define RCLCPP_DEBUG(...) ((void)0)
namespace cost_value { constexpr unsigned char FREE_SPACE=0, NO_INFORMATION=128, LETHAL_OBSTACLE=255; }
struct PointCloud2 { std::vector<float> x,y; };
struct Pose { struct { double x,y; } position; };
template<class T> class PointCloud2ConstIterator {
 const std::vector<float>* p; size_t i;
 public:
 PointCloud2ConstIterator(const PointCloud2& v,const char* n): p(n[0]=='x'?&v.x:&v.y),i(0){}
 PointCloud2ConstIterator end() const { auto c=*this; c.i=p->size(); return c; }
 bool operator!=(const PointCloud2ConstIterator& b)const{return i!=b.i;}
 T operator*()const{return (*p)[i];}
 void operator++(){++i;}
};
class OccupancyGridMap {
 public:
 unsigned int size_x_,size_y_; double resolution_,origin_x_,origin_y_;
 std::vector<unsigned char> data; unsigned char* costmap_;
 OccupancyGridMap(unsigned int x,unsigned int y,float r,double ox,double oy):size_x_(x),size_y_(y),resolution_(r),origin_x_(ox),origin_y_(oy),data(x*y,128),costmap_(data.data()){}
 unsigned int getIndex(unsigned int x,unsigned int y)const{return y*size_x_+x;}
 bool worldToMap(double,double,unsigned int&,unsigned int&)const;
 void raytrace2D(const PointCloud2&,const Pose&);
 void raytraceFreespace(const PointCloud2&,const Pose&);
 void updateFreespaceCells(const PointCloud2&);
 void updateOccupiedCells(const PointCloud2&);
 void updateCellsByPointCloud(const PointCloud2&,unsigned char);
 class MarkCell {unsigned char* p;unsigned char v;public:MarkCell(unsigned char* p_,unsigned char v_):p(p_),v(v_){}void operator()(unsigned int i){p[i]=v;}};
'''
    result = preamble
    for sig in ['inline void raytraceLine(', 'inline void bresenham2D(']:
        result += '\ntemplate<class ActionType>\n' + function(nav2, sig)
    result += '\n' + function(nav2, 'inline int sign(') + '\n};\n'
    for sig in ['bool OccupancyGridMap::worldToMap(', 'void OccupancyGridMap::raytrace2D(',
                'void OccupancyGridMap::raytraceFreespace(', 'void OccupancyGridMap::updateFreespaceCells(',
                'void OccupancyGridMap::updateOccupiedCells(', 'void OccupancyGridMap::updateCellsByPointCloud(']:
        result += function(source, sig) + '\n'
    result += r'''
struct Index {enum {FREE=0,OCCUPIED=1};};
class OccupancyGridMapBBFUpdater {
 struct Matrix {double operator()(int row,int col)const {double a[2][2]={{.8,.05},{.2,.95}};return a[row][col];}} probability_matrix_;
 double v_ratio_=.1;
 public: unsigned char applyBBF(const unsigned char&,const unsigned char&);
};
'''
    result += function(bayes, 'inline unsigned char OccupancyGridMapBBFUpdater::applyBBF(')
    result += r'''
int main(){
 std::string mode; std::cin>>mode;
 if(mode=="bbf") {OccupancyGridMapBBFUpdater b; for(int z: {0,128,255})for(int o=0;o<256;o++)std::cout<<int(b.applyBBF(z,o))<<" ";return 0;}
 unsigned int nx,ny,n;float r;double ox,oy,sx,sy;
 std::cin>>nx>>ny>>r>>ox>>oy>>sx>>sy>>n;
 OccupancyGridMap g(nx,ny,r,ox,oy);PointCloud2 p;
 for(unsigned int i=0;i<n;i++){float x,y;std::cin>>x>>y;p.x.push_back(x);p.y.push_back(y);}
 Pose pose;pose.position.x=sx;pose.position.y=sy;g.raytrace2D(p,pose);
 for(auto v:g.data)std::cout<<int(v)<<" ";
}
'''
    return result


class CppParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which('g++'):
            raise unittest.SkipTest('g++ required for independent upstream parity check')
        cls.temp = tempfile.TemporaryDirectory()
        source = Path(cls.temp.name) / 'oracle.cpp'
        source.write_text(cpp_source())
        cls.exe = source.with_suffix('')
        subprocess.run(['g++', '-std=c++17', '-O2', '-ffp-contract=off', str(source), '-o', str(cls.exe)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'temp'):
            cls.temp.cleanup()

    def test_all_bbf_byte_inputs(self):
        output = subprocess.check_output([str(self.exe)], input='bbf\n', text=True)
        expected = np.fromstring(output, sep=' ', dtype=np.uint8).reshape(3, 256)
        actual = apply_bbf(np.array([0, 128, 255])[:, None], np.arange(256)[None, :])
        np.testing.assert_array_equal(actual, expected)

    def test_randomized_and_boundary_rays(self):
        rng = np.random.default_rng(13)
        for case in range(80):
            nx, ny = 32, 29
            res = [1., .4, .1, .5][case % 4]
            ox, oy = -4.25, -5.5
            r = float(np.float32(res))
            sensor = (ox + 9.5*r, oy + 10.5*r) if case % 5 else (ox - .1, oy)
            points = rng.uniform([-10, -10], [nx+10, ny+10], size=(case % 30, 2)) * r + [ox, oy]
            boundaries = np.array([[ox, oy], [ox+nx*r, oy+10*r], [ox+10*r, oy+ny*r], sensor])
            points = np.concatenate([points, boundaries]).astype(np.float32)
            request = f'map {nx} {ny} {res} {ox} {oy} {sensor[0]} {sensor[1]} {len(points)}\n'
            request += '\n'.join(f'{float(x)} {float(y)}' for x,y in points)
            expected = np.fromstring(subprocess.check_output([str(self.exe)], input=request, text=True), sep=' ', dtype=np.uint8).reshape(ny, nx)
            grid = OccupancyGridMap(nx, ny, res)
            grid.origin_x_, grid.origin_y_ = ox, oy
            grid.raytrace2D(points, sensor)
            np.testing.assert_array_equal(grid.costmap_, expected, err_msg=f'case {case}')

    def test_origin_shift_and_first_reset(self):
        grid = OccupancyGridMap(8, 8, 1.)
        grid.costmap_[2, 3] = 255
        grid.updateOrigin(0, 0)
        self.assertTrue(np.all(grid.costmap_ == 128))
        grid.costmap_[2, 3] = 255
        grid.updateOrigin(1, -1)
        self.assertEqual(grid.costmap_[3, 2], 255)
        grid.updateOrigin(100, 100)
        self.assertTrue(np.all(grid.costmap_ == 128))

    def test_vendored_source_hashes(self):
        import hashlib
        manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
        for source in manifest['files']:
            self.assertEqual(hashlib.sha256((ROOT / source['local_path']).read_bytes()).hexdigest(), source['sha256'])


if __name__ == '__main__':
    unittest.main()

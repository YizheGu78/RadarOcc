"""Execute the actual CUDA scalar helper as host C++, without a CUDA driver.

This verifies interpolation math, not GPU launches or parallel reductions.
"""
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
KERNEL=ROOT/'third_party/VoxFormer/deform_attn_3d/csrc/ms_deform_attn_cuda_kernel.cuh'


class CudaScalarGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler=shutil.which('g++') or shutil.which('c++')
        if compiler is None:raise unittest.SkipTest('Host C++ compiler unavailable')
        cls.temp=tempfile.TemporaryDirectory(prefix='radarocc-cuda-math-')
        cls.addClassCleanup(cls.temp.cleanup)
        source=KERNEL.read_text()
        # Extract all three unmodified scalar helpers, excluding CUDA headers,
        # __global__ launches and reductions; only __device__ becomes ordinary C++.
        start=source.index('template <typename scalar_t>')
        end=source.index('template <typename scalar_t>\n__global__')
        helpers=source[start:end].replace('__device__','')
        preamble='#include <cmath>\ninline void atomicAdd(double* p,double v){*p+=v;}\n'
        bridge=r'''
extern "C" void run(const double* data,int height,int width,int depth,
                    int heads,int channels,double h,double w,double z,
                    int head,int channel,double top,double weight,int use_atomic,
                    double* output,double* gv,double* gl,double* ga) {
  const double* ptr=data;
  *output=ms_deform_attn_im2col_bilinear(ptr,height,width,depth,heads,channels,h,w,z,head,channel)*weight;
  if(use_atomic) ms_deform_attn_col2im_bilinear_gm(ptr,height,width,depth,heads,channels,h,w,z,head,channel,top,weight,gv,gl,ga);
  else ms_deform_attn_col2im_bilinear(ptr,height,width,depth,heads,channels,h,w,z,head,channel,top,weight,gv,gl,ga);
}
'''
        folder=Path(cls.temp.name)
        cpp=folder/'helpers.cpp';cpp.write_text(preamble+helpers+bridge)
        so=folder/'helpers.so'
        subprocess.run([compiler,'-O2','-shared','-fPIC',str(cpp),'-o',str(so)],
                       check=True,capture_output=True,text=True)
        cls.library=ctypes.CDLL(str(so))
        cls.pointer=ctypes.POINTER(ctypes.c_double)
        cls.library.run.argtypes=[cls.pointer]+[ctypes.c_int]*5+[ctypes.c_double]*3+[
            ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_int]+[cls.pointer]*4

    def test_actual_both_backward_helpers_match_autograd_and_finite_differences(self):
        rng=np.random.default_rng(18)
        # Includes interior and partial zero padding on every axis.
        points=[(1.3,2.2,.7),(-.2,1.4,.8),(2.3,1.6,.4),
                (.4,-.3,1.2),(1.1,4.2,.8),(.7,2.3,-.2),(1.6,1.2,1.3)]
        for atomic in (False,True):
            for point in points:
                with self.subTest(atomic=atomic,point=point):
                    a,r,e,heads,channels=3,5,2,2,3
                    storage=rng.normal(size=(a,r,e,heads,channels)).astype(np.float64)
                    value_grad=np.zeros_like(storage)
                    location_grad=np.zeros(3,dtype=np.float64)
                    attention_grad=np.zeros(1,dtype=np.float64)
                    output=np.zeros(1,dtype=np.float64)
                    head,channel=1,2
                    top,weight=.7,.4
                    arrays=[np.ascontiguousarray(x) for x in
                            (storage,output,value_grad,location_grad,attention_grad)]
                    data,output,value_grad,location_grad,attention_grad=arrays
                    self.library.run(data.ctypes.data_as(self.pointer),a,r,e,heads,channels,
                        *point,head,channel,top,weight,int(atomic),
                        *[x.ctypes.data_as(self.pointer) for x in arrays[1:]])
                    field=torch.from_numpy(data).requires_grad_()
                    # Storage [A,R,E,H,C] -> grid_sample [1,H*C,E,A,R].
                    volume=field.permute(3,4,2,0,1).reshape(1,heads*channels,e,a,r)
                    location=(torch.tensor(point,dtype=torch.float64)[[1,0,2]]+.5)/torch.tensor([r,a,e])
                    location.requires_grad_()
                    attention=torch.tensor(weight,dtype=torch.float64,requires_grad=True)
                    def sample(loc):
                        grid=(loc*2-1).reshape(1,1,1,1,3)
                        return F.grid_sample(volume,grid,align_corners=False)[0,head*channels+channel,0,0,0]*attention
                    expected=sample(location)
                    expected.mul(top).backward()
                    np.testing.assert_allclose(output,float(expected.detach()),atol=1e-12)
                    np.testing.assert_allclose(value_grad,field.grad.numpy(),atol=1e-12)
                    np.testing.assert_allclose(location_grad,location.grad.numpy(),atol=1e-11)
                    np.testing.assert_allclose(attention_grad,float(attention.grad),atol=1e-12)
                    # Independent central differences in normalized [R,A,E].
                    numerical=[]
                    for axis in range(3):
                        delta=torch.zeros(3,dtype=torch.float64);delta[axis]=1e-6
                        numerical.append(float(((sample(location.detach()+delta)-sample(location.detach()-delta))*top/(2e-6)).detach()))
                    np.testing.assert_allclose(location_grad,numerical,rtol=1e-8,atol=1e-9)


if __name__=='__main__':unittest.main()

"""Compare Python against a compiled, unmodified copy of the user's OctoMap fork."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from tradition_real.octomap.octree import OcTree

ROOT = Path(__file__).resolve().parents[1]


class OctomapCppParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which('g++') is None:
            raise unittest.SkipTest('g++ required for original OctoMap C++ parity')
        cls.folder = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.folder.name) / 'octomap_reference'
        vendor = ROOT / 'vendor/octomap'
        sources = [vendor/'src'/name for name in (
            'AbstractOcTree.cpp', 'AbstractOccupancyOcTree.cpp', 'OcTree.cpp',
            'OcTreeNode.cpp', 'Pointcloud.cpp', 'ScanGraph.cpp',
            'math/Pose6D.cpp', 'math/Quaternion.cpp', 'math/Vector3.cpp')]
        subprocess.run(['g++', '-std=c++11', '-O2', '-I', str(vendor/'include'),
                        str(ROOT/'tests/octomap_reference.cpp'), *map(str,sources),
                        '-o', str(cls.binary)], check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def compare(self, operations, parameters=(.4,.7,.4,.5,.1192,.971)):
        tree = OcTree(parameters[0])
        for setter,value in zip(('setProbHit','setProbMiss','setOccupancyThres','setClampingThresMin','setClampingThresMax'), parameters[1:]):
            getattr(tree,setter)(value)
        payload = [' '.join(map(str, parameters))]
        expected = []
        for op,args in operations:
            payload.append(op + ' ' + ' '.join(map(str,args)))
            if op == 'R':
                ray = tree.computeRayKeys(args[:3], args[3:])
                expected.append(None if ray is None else [list(k) for k in ray])
            elif op == 'S':
                count, maximum, lazy, discrete = args[:4]
                tree.insertPointCloud(np.asarray(args[7:]).reshape(int(count),3),args[4:7],maximum,bool(lazy),bool(discrete))
            elif op == 'U': tree.updateNode(tuple(args[:3]),float(args[3]),bool(args[4]))
            elif op == 'Q':
                node=tree.search(tuple(args[:3]),args[3])
                expected.append(None if node is None else [node.value,int(tree.isNodeOccupied(node))])
            elif op == 'I': tree.updateInnerOccupancy()
            elif op == 'P': tree.prune()
            elif op == 'C': tree.clear()
            elif op == 'B':
                tree.setBBXMin(args[:3]); tree.setBBXMax(args[3:6]); tree.useBBXLimit(args[6])
            elif op == 'D':
                expected.append({'size':tree.size(),'root':tree.root.value if tree.root else None,
                                 'leaves': sorted([*lo,span,node.value] for lo,span,node in tree.iter_leaves())})
        result=subprocess.run([str(self.binary)],input='\n'.join(payload)+'\n',capture_output=True,text=True,check=True)
        actual=[json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(expected),len(actual))
        for index,(python,cpp) in enumerate(zip(expected,actual)):
            self.assertEqual(python,cpp,f'Output {index}: Python/C++ mismatch')
        return tree

    def test_original_sources_match_manifest(self):
        manifest=json.loads((ROOT/'OCTOMAP_SOURCE_MANIFEST.json').read_text())
        for entry in manifest['files']:
            self.assertEqual(hashlib.sha256((ROOT/entry['local_path']).read_bytes()).hexdigest(),entry['sha256'])

    def test_3d_rays_random_faces_edges_corners_and_bounds(self):
        rng=np.random.default_rng(51)
        pairs=[([0,0,0],[4,4,4]),([0,0,0],[-4,-4,-4]),
               ([.01,.01,.01],[.39,.39,.39]),([0,0,0],[0,0,0]),
               ([0,0,0],[4,0,0]),([0,0,0],[0,-4,0]),([0,0,0],[0,0,4]),
               ([.4,.4,.4],[0,0,0]),([13108,0,0],[0,0,0])]
        pairs += [(rng.uniform(-8,8,3),rng.uniform(-8,8,3)) for _ in range(100)]
        pairs += [(rng.integers(-10,10,3)*.4,rng.integers(-10,10,3)*.4) for _ in range(40)]
        self.compare([('R',[*a,*b]) for a,b in pairs])

    def test_scan_dedup_hit_priority_maxrange_discrete_and_bbx(self):
        rng=np.random.default_rng(4)
        for lazy,discrete,maximum,bbx in [(0,0,-1,False),(1,0,-1,False),(0,1,-1,False),
                                         (0,0,2.5,False),(1,1,2.5,False),(0,0,0,False),
                                         (0,0,2.5,True),(0,1,-1,True)]:
            with self.subTest(lazy=lazy,discrete=discrete,maximum=maximum,bbx=bbx):
                ops=[]
                if bbx: ops.append(('B',[-2,-2,-2,2,2,2,1]))
                for frame in range(5):
                    cloud=rng.uniform(-4,4,(18,3))
                    cloud=np.vstack([cloud,[[.8,.01,.01],[.8,.01,.01],[2,.01,.01],[0,0,0]]])
                    origin=[0,0,0] if frame < 4 else [-2.5,0,0]
                    ops += [('S',[len(cloud),maximum,lazy,discrete,*origin,*cloud.flatten()]),('D',[])]
                ops += [('I',[]),('P',[]),('D',[])]
                self.compare(ops)

    def test_float_clamping_threshold_parent_prune_expand_and_lazy(self):
        keys=[(32768+x,32768+y,32768+z) for x in range(2) for y in range(2) for z in range(2)]
        ops=[]
        for lazy in [0,1]:
            ops.append(('C',[]))
            for k in keys: ops.append(('U',[*k,.8472978472709656,lazy]))
            ops += [('D',[]),('Q',[*keys[0],15]),('I',[]),('P',[]),('D',[])]
            # Expand a pruned parent, clamp both directions, then recover.
            for delta,count in [(.8472978472709656,12),(-.40546509623527527,22),(.8472978472709656,9)]:
                for _ in range(count):
                    ops += [('U',[*keys[0],delta,lazy]),('Q',[*keys[0],16])]
                ops += [('I',[]),('D',[])]
        ops += [('C',[]),('U',[32768,32768,32768,0.,0]),('Q',[32768,32768,32768,0]),('D',[])]
        self.compare(ops)
        self.compare(ops,parameters=(.2,.8,.3,.6,.1,.98))


if __name__ == '__main__':
    unittest.main()

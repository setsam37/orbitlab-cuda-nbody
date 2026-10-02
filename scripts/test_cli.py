"""Integration checks run with ORBITLAB_EXE pointing to the built CLI."""
import csv, json, math, os, subprocess, tempfile, unittest
from pathlib import Path

@unittest.skipUnless(os.environ.get('ORBITLAB_EXE'), 'Set ORBITLAB_EXE to run CLI integration tests')
class CliTests(unittest.TestCase):
    def run_cli(self,*args):
        return subprocess.run([os.environ['ORBITLAB_EXE'],*map(str,args)],capture_output=True,text=True)
    def test_invalid_arguments_fail(self):
        for args in [('simulate','--n','0'),('simulate','--dt','nan'),('simulate','--dt','0'),('simulate','--steps','-1'),('simulate','--n',str(2**64-1)),('simulate','--wat','1'),('simulate','--precision','half'),('simulate','--backend','bogus')]:
            with self.subTest(args=args):
                p=self.run_cli(*args);self.assertNotEqual(p.returncode,0);self.assertIn('ERROR',p.stderr)
    def test_exact_time_and_paths_with_spaces(self):
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'new parent'/'orbit results'
            p=self.run_cli('simulate','--precision','double','--steps','7','--dt','.03','--sample-every','3','--output',out)
            self.assertEqual(p.returncode,0,p.stderr)
            rows=list(csv.DictReader((out/'trajectory.csv').open()))
            self.assertEqual(sorted(set(int(r['step']) for r in rows)),[0,3,6,7])
            self.assertEqual(float(rows[-1]['time']),7*.03)
            self.assertEqual(json.loads((out/'metadata.json').read_text())['steps'],7)
    def test_zero_steps_outputs_initial_state(self):
        with tempfile.TemporaryDirectory() as root:
            p=self.run_cli('simulate','--steps','0','--output',root)
            self.assertEqual(p.returncode,0,p.stderr)
            rows=list(csv.DictReader((Path(root)/'trajectory.csv').open()));self.assertEqual(len(rows),2)
            self.assertTrue(all(r['step']=='0' for r in rows))
    def test_unwritable_destination_fails(self):
        with tempfile.TemporaryDirectory() as root:
            blocker=Path(root)/'file';blocker.write_text('occupied')
            p=self.run_cli('simulate','--output',blocker/'child')
            self.assertNotEqual(p.returncode,0);self.assertIn('ERROR',p.stderr)
    def test_initial_state_overrides_fixture(self):
        with tempfile.TemporaryDirectory() as root:
            initial=Path(root)/'initial.csv';initial.write_text('id,mass,x,y,z,vx,vy,vz\n0,1,0,0,0,1,0,0\n')
            out=Path(root)/'out';p=self.run_cli('simulate','--initial-state',initial,'--steps','10','--dt','.1','--precision','double','--output',out)
            self.assertEqual(p.returncode,0,p.stderr)
            rows=list(csv.DictReader((out/'trajectory.csv').open()));self.assertAlmostEqual(float(rows[-1]['x']),1)
            self.assertEqual(json.loads((out/'metadata.json').read_text())['n'],1)
    def test_benchmark_produces_raw_batches_and_configuration(self):
        with tempfile.TemporaryDirectory() as root:
            p=self.run_cli('benchmark','--n','17','--mode','force','--precision','double','--output',root)
            self.assertEqual(p.returncode,0,p.stderr)
            rows=list(csv.DictReader((Path(root)/'batches.csv').open()));self.assertEqual(len(rows),5)
            self.assertTrue(all(float(r['per_call_ms'])>0 and int(r['repetitions'])==5 for r in rows))
            meta=json.loads((Path(root)/'metadata.json').read_text());self.assertEqual(meta['precision'],'double');self.assertEqual(meta['mode'],'force')

if __name__=='__main__': unittest.main()

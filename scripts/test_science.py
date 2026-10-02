import math,tempfile,unittest
from pathlib import Path
from science import convergence_ratios,invariant_summary,load_numeric_csv
class ScienceTests(unittest.TestCase):
    def test_second_order_ratios(self):
        self.assertEqual(convergence_ratios([.16,.04,.01]),[4,4])
    def test_invalid_errors_rejected(self):
        for errors in ([1,0],[math.nan,1],[-1,1]):
            with self.subTest(errors=errors):
                with self.assertRaises(ValueError):convergence_ratios(errors)
    def test_invariant_scales_handle_zero_initial_momentum(self):
        rows=[dict(energy=-2,px=0,py=0,pz=0,lx=0,ly=0,lz=4),dict(energy=-1.8,px=.3,py=.4,pz=0,lx=0,ly=0,lz=4.2)]
        s=invariant_summary(rows,2,2,4)
        self.assertAlmostEqual(s['max_energy'],.1);self.assertAlmostEqual(s['max_momentum'],.25);self.assertAlmostEqual(s['max_angular_momentum'],.05)
    def test_zero_scales_rejected(self):
        with self.assertRaises(ValueError):invariant_summary([],0,1,1)
    def test_nonfinite_data_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'bad.csv';p.write_text('x,y\n1,nan\n')
            with self.assertRaises(ValueError):load_numeric_csv(p)
    def test_valid_numeric_csv(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'good.csv';p.write_text('x,y\n1,2\n')
            self.assertEqual(load_numeric_csv(p),[dict(x=1.,y=2.)])
if __name__=='__main__':unittest.main()

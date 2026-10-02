import math,unittest
from benchmarks import summarize,speedup
class BenchmarkTests(unittest.TestCase):
    def test_directed_interactions_and_median(self):
        s=summarize([900,1000,1100],2,'force')
        self.assertEqual(s['median_ms'],1000);self.assertEqual(s['min_ms'],900);self.assertEqual(s['max_ms'],1100);self.assertEqual(s['interactions_per_second'],2)
    def test_step_does_not_claim_force_throughput(self):
        self.assertIsNone(summarize([1],17,'step')['interactions_per_second'])
    def test_rejects_invalid_times(self):
        for values in ([],[0],[-1],[math.nan],[math.inf]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):summarize(values,2,'force')
    def test_speedup_requires_compatible_workloads(self):
        cpu=dict(n=17,precision='float',mode='force',median_ms=10)
        gpu=dict(n=17,precision='float',mode='force',median_ms=2)
        self.assertEqual(speedup(cpu,gpu),5)
        for key,value in [('n',18),('precision','double'),('mode','step'),('median_ms',0)]:
            bad=gpu.copy();bad[key]=value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):speedup(cpu,bad)
if __name__=='__main__':unittest.main()

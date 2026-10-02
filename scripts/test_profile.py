import sys,unittest,runpy,subprocess
from unittest.mock import patch
from pathlib import Path
from profile_kernels import profile
class ProfileTests(unittest.TestCase):
    def test_missing_tool_is_unavailable(self):
        r=profile('unused','unused',['orbitlab-nonexistent-profile-tool']);self.assertEqual(r['status'],'unavailable')
    def test_counter_denial_is_reported(self):
        cmd=[sys.executable,'-c','import sys;sys.stderr.write("ERR_NVGPUCTRPERM: permission denied");sys.exit(1)']
        r=profile('unused','unused',cmd);self.assertEqual(r['status'],'unavailable');self.assertIn('ERR_NVGPUCTRPERM',r['stderr'])
    def test_other_failures_remain_failures(self):
        r=profile('unused','unused',[sys.executable,'-c','import sys;sys.exit(7)']);self.assertEqual(r['status'],'failed');self.assertEqual(r['exit_code'],7)
    def test_cli_exits_nonzero_and_preserves_failure_logs(self):
        with patch.object(sys,'argv',['profile_kernels.py','--exe','unused','--output','unused']), patch('subprocess.run',return_value=subprocess.CompletedProcess([],7,'','real failure')), patch.object(Path,'mkdir'), patch.object(Path,'write_text') as write:
            with self.assertRaises(SystemExit) as exit:
                runpy.run_module('profile_kernels',run_name='__main__')
            self.assertNotEqual(exit.exception.code,0)
            self.assertEqual(write.call_count,2)
            self.assertIn('real failure',write.call_args.args[0])
if __name__=='__main__':unittest.main()

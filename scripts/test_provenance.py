import unittest
from provenance import parse_cache

class ProvenanceTests(unittest.TestCase):
    def test_actual_build_configuration(self):
        info=parse_cache('CMAKE_BUILD_TYPE:STRING=Debug\nCMAKE_CXX_COMPILER:FILEPATH=/custom/clang++\nCMAKE_CUDA_FLAGS:STRING=--use_fast_math\n')
        self.assertEqual(info['build_type'],'Debug')
        self.assertEqual(info['cxx_compiler'],'/custom/clang++')
        self.assertIn('--use_fast_math',info['configured_flags']['CMAKE_CUDA_FLAGS'])
    def test_unknown_is_explicit(self):
        info=parse_cache('unrelated:BOOL=ON\n')
        self.assertIsNone(info['build_type'])
        self.assertIsNone(info['cxx_compiler'])

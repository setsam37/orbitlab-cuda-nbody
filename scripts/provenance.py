"""Record observed build artifacts; absence means unknown, not a guessed configuration."""
import hashlib
from pathlib import Path

def parse_cache(text):
    values={}
    for line in text.splitlines():
        if line.startswith(('#','//')) or ':' not in line or '=' not in line:continue
        key,rest=line.split(':',1);values[key]=rest.split('=',1)[1]
    return dict(build_type=values.get('CMAKE_BUILD_TYPE'),
                cxx_compiler=values.get('CMAKE_CXX_COMPILER'),
                cuda_compiler=values.get('CMAKE_CUDA_COMPILER'),
                cuda_architectures=values.get('CMAKE_CUDA_ARCHITECTURES'),
                configured_flags={k:v for k,v in values.items() if k.startswith(('CMAKE_CXX_FLAGS','CMAKE_CUDA_FLAGS'))})

def observe_build(exe):
    exe=Path(exe).resolve();cache=exe.parent/'CMakeCache.txt'
    info=parse_cache(cache.read_text() if cache.exists() else '')
    info.update(cache_available=cache.exists(),executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                generated_flags={str(p.relative_to(exe.parent)):p.read_text() for p in (exe.parent/'CMakeFiles').glob('*/flags.make')},
                provenance_scope='Observed adjacent build files and binary hash; current source hashes do not prove binary/source correspondence.')
    return info

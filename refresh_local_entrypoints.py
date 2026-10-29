"""Regenerate copied environment console launchers for the active local Python."""
import importlib.metadata
import os
import sys
from pathlib import Path
from pip._vendor.distlib.scripts import ScriptMaker

ROOT = Path(__file__).resolve().parent
env = Path(sys.prefix).resolve()
if not env.is_relative_to(ROOT):
    raise SystemExit('Run using an environment inside this project')
scripts = env/'Scripts'
maker = ScriptMaker(None, str(scripts))
maker.executable = str(Path(sys.executable).resolve())
maker.clobber = True
maker.variants = {''}
count = 0
for dist in importlib.metadata.distributions():
    for entry in dist.entry_points:
        if entry.group not in ('console_scripts','gui_scripts'):
            continue
        # Detach existing linked launcher before distlib modifies it.
        for suffix in ('.exe','.py','-script.py'):
            file = scripts/(entry.name+suffix)
            if file.exists():
                temp = file.with_name(file.name+'.detach')
                temp.write_bytes(file.read_bytes())
                os.replace(temp,file)
        maker.make(f'{entry.name} = {entry.value}', options={'gui':entry.group=='gui_scripts'})
        count += 1
print(f'Regenerated {count} entry points in {scripts}')

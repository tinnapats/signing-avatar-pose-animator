"""Temporarily hide precisely named legacy roots, test, and restore in finally.

Requires --allow-temporary-rename and appropriate filesystem permission.
Never deletes legacy roots. Run only when no legacy application is using them.
"""
import argparse
import json
import os
import socket
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-temporary-rename', action='store_true', required=True)
    args = parser.parse_args()
    archive_root = Path.home()/'.codex/visualizations/2026/07/27/019fa278-2887-7562-a831-6544ada9c461'
    legacy = [ROOT.parent/'signing-avatar-github-tinnapat', ROOT.parent/'signing-avatar-pose-animator',
              archive_root/'model-results', archive_root/'s310']
    out = ROOT/'local_archive'/('isolation-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    moved, checks, server = [], [], None
    env = os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','SIGN_PYTHON','SIGN_EVAL_PYTHON','FIDELITY_SOURCE','FIDELITY_OUTPUT'):
        env.pop(key,None)
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'
    env['AUDIT_BROWSER'] = str(ROOT/'local_runtime/chrome/chrome.exe')
    env['FIDELITY_OUTPUT'] = str(out/'fidelity')
    py = str(ROOT/'.venv/Scripts/python.exe')
    def run(name, command, timeout=300):
        print('TEST', name, flush=True)
        with (out/(name+'.log')).open('wb') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW)
        checks.append(dict(test=name, code=result.returncode))
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {out/name}.log')
    try:
        with socket.socket() as probe:
            probe.bind(('127.0.0.1',8043))
        for source in legacy:
            target = source.with_name(source.name+'.isolation-backup')
            if not source.is_dir() or target.exists():
                raise RuntimeError(f'Cannot safely hide {source}')
            source.rename(target)
            moved.append((source,target))
        assert not any(p.exists() for p in legacy)
        print('All four legacy paths are temporarily unavailable.', flush=True)
        run('assets', [py,'-B','check_friend_setup.py'])
        run('unit-tests', [py,'-B','-m','unittest','discover','-s','tests','-q'])
        with (out/'server.log').open('wb') as log:
            server = subprocess.Popen([py,'-B','-u','run_pose_animator_server.py','--port','8043','--no-browser'],
                                      cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                      creationflags=subprocess.CREATE_NO_WINDOW)
        health = None
        for _ in range(60):
            if server.poll() is not None:
                raise RuntimeError('Server exited')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8043/api/health',timeout=2) as response:
                    health = json.load(response)
                if health['stt']['loaded']:
                    break
            except OSError:
                pass
            time.sleep(1)
        assert health and health['stt']['loaded'], health
        assert Path(health['dataDir']).resolve() == ROOT/'SLclean.sqlite3'
        assert Path(health['stt']['model']).resolve().is_relative_to(ROOT/'models')
        with urllib.request.urlopen('http://127.0.0.1:8043/dataset_player.html') as response:
            assert response.status == 200
        frames = {}
        for word in ('hello','love','abdomen'):
            with urllib.request.urlopen('http://127.0.0.1:8043/api/generate_sequence?text='+word,timeout=90) as response:
                data = json.load(response)
            assert data['ok'] and data['payload']['frames']
            frames[word] = len(data['payload']['frames'])
        checks.append(dict(test='server-local-model-and-sqlite', frames=frames, code=0))
        for experiment, model in [(1,'stgcn'),(1,'slgcn'),(2,'stgcn'),(3,'stgcn')]:
            run(f'exp{experiment}-{model}', [py,'-B','-X','utf8','evaluation/run_evaluation.py','--experiment',str(experiment),'--model',model,'--limit','200'])
        run('rtmw-image', [str(ROOT/'recognition_audit/rtmpose_env/Scripts/python.exe'),'-B','evaluation/check_image_runtime.py'])
        run('mediapipe-image', [str(ROOT/'recognition_audit/paired_video_env/Scripts/python.exe'),'-B','verify_local_runtimes.py','mediapipe'])
        run('sdpose-head', [str(ROOT/'local_runtime/sdpose_env/Scripts/python.exe'),'-B','verify_local_runtimes.py','sdpose'])
        # Keep legacy fidelity inputs unchanged; save/restore even on failure.
        inputs = ROOT/'fidelity_results/inputs.json'
        original = inputs.read_bytes() if inputs.exists() else None
        try:
            run('fidelity-prepare',[py,'-B','measure_render_fidelity.py','prepare'])
            run('fidelity-canvas',[str(ROOT/'local_runtime/node/node.exe'),'verify_render_fidelity.mjs'])
            run('fidelity-summary',[py,'-B','summarize_fidelity_verification.py'])
        finally:
            if original is not None:
                temporary = inputs.with_suffix('.restore-tmp')
                temporary.write_bytes(original)
                os.replace(temporary, inputs)
        assert not any(p.exists() for p in legacy)
    finally:
        restore_errors = []
        if server:
            try:
                server.terminate()
                server.wait(timeout=30)
            except OSError as error:
                restore_errors.append(str(error))
            except subprocess.TimeoutExpired:
                server.kill()
        for source,target in reversed(moved):
            try:
                target.rename(source)
            except OSError as error:
                restore_errors.append(str(error))
        (out/'results.json').write_text(json.dumps(dict(checks=checks,restore_errors=restore_errors),indent=2),encoding='utf-8')
        print('REPORT',out,flush=True)
        if restore_errors:
            raise RuntimeError(restore_errors)
        print('Legacy directory names restored; no originals deleted.',flush=True)


if __name__=='__main__':
    main()

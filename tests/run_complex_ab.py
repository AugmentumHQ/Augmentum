"""Run complex builds that stretch beyond reference coverage."""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

from augmentum.tools import application_references
from augmentum.tools.artifact_application import ApplicationBuilderTool

MODEL = ''
try:
    resp = urllib.request.urlopen('http://127.0.0.1:1234/v1/models', timeout=5)
    MODEL = json.loads(resp.read())['data'][0]['id']
except: pass

async def llm(messages, max_tokens=4096, model='', **kw):
    payload = json.dumps({'model': MODEL, 'messages': messages, 'max_tokens': max(max_tokens, 4096), 'stream': False}).encode()
    req = urllib.request.Request('http://127.0.0.1:1234/v1/chat/completions', data=payload, headers={'Content-Type': 'application/json'})
    resp = urllib.request.urlopen(req, timeout=300)
    return json.loads(resp.read())['choices'][0]['message'].get('content', '')

class S:
    async def save(self, **kw): return {'id': 'x'}

orig_select = application_references.select_references

builds = [
    {
        "desc": "physics sandbox with bouncing balls that have gravity, elasticity, and mouse drag. Spawn balls by clicking, adjust gravity with a slider.",
        "scaffold": "game",
        "checks": {
            'Canvas + getContext': lambda js, code: 'getContext' in js,
            'Gravity constant': lambda js, code: 'gravity' in js.lower() or 'GRAVITY' in js,
            'Velocity (vx/vy)': lambda js, code: 'vx' in js or 'velocity' in js.lower(),
            'Elastic collision/bounce': lambda js, code: 'bounce' in js.lower() or 'elastic' in js.lower() or 'restitution' in js.lower(),
            'Mouse click spawn': lambda js, code: 'click' in js and ('push' in js or 'spawn' in js.lower()),
            'Mouse drag': lambda js, code: 'mousedown' in js and 'mousemove' in js,
            'Slider/range control': lambda js, code: 'range' in code.lower() or 'slider' in code.lower(),
            'requestAnimationFrame': lambda js, code: 'requestAnimationFrame' in js,
            'Ball-ball collision': lambda js, code: 'sqrt' in js or 'hypot' in js or 'dist' in js.lower(),
            'Wall bounce': lambda js, code: '*= -' in js or '*=-' in js or 'bounce' in js.lower(),
        }
    },
    {
        "desc": "music visualizer that uses Web Audio API to analyze microphone input and draws animated frequency bars and waveform on canvas",
        "scaffold": "game",
        "checks": {
            'AudioContext': lambda js, code: 'AudioContext' in js,
            'getUserMedia': lambda js, code: 'getUserMedia' in js or 'mediaDevices' in js,
            'AnalyserNode': lambda js, code: 'createAnalyser' in js or 'analyser' in js.lower(),
            'FrequencyData': lambda js, code: 'FrequencyData' in js or 'frequencyData' in js,
            'Canvas drawing': lambda js, code: 'fillRect' in js or 'lineTo' in js,
            'requestAnimationFrame': lambda js, code: 'requestAnimationFrame' in js,
            'Bar visualization': lambda js, code: 'bar' in js.lower() and 'width' in js,
            'Dynamic color': lambda js, code: 'hsl' in js.lower() or 'rgb(' in js.lower(),
        }
    },
    {
        "desc": "collaborative whiteboard with drawing tools (pen, rectangle, circle, eraser), color picker, undo/redo, and export to PNG",
        "scaffold": "static",
        "checks": {
            'Canvas drawing': lambda js, code: 'getContext' in js and 'lineTo' in js,
            'Pen/freehand tool': lambda js, code: 'pen' in js.lower() or 'freehand' in js.lower() or ('mousedown' in js and 'lineTo' in js),
            'Rectangle tool': lambda js, code: 'rect' in js.lower(),
            'Circle tool': lambda js, code: 'circle' in js.lower() or 'arc(' in js,
            'Eraser': lambda js, code: 'eraser' in js.lower() or 'erase' in js.lower(),
            'Color picker': lambda js, code: 'color' in code.lower() and 'type="color"' in code.lower(),
            'Undo/redo': lambda js, code: 'undo' in js.lower(),
            'Export PNG': lambda js, code: 'toDataURL' in js or 'toBlob' in js,
            'Mouse events': lambda js, code: 'mousedown' in js and 'mousemove' in js,
        }
    },
    {
        "desc": "markdown note-taking app with live preview, sidebar with multiple notes, search filter, and localStorage persistence",
        "scaffold": "form",
        "checks": {
            'Markdown parsing': lambda js, code: '##' in js or 'replace(' in js and ('h1' in js.lower() or 'strong' in js.lower()),
            'Live preview on input': lambda js, code: 'input' in js and 'preview' in js.lower(),
            'Multiple notes array': lambda js, code: 'notes' in js.lower() and ('[]' in js or 'Array' in js or 'push' in js),
            'Sidebar/list UI': lambda js, code: 'sidebar' in code.lower() or 'note-list' in code.lower(),
            'Search/filter': lambda js, code: 'search' in js.lower() or 'filter' in js.lower(),
            'localStorage': lambda js, code: 'localStorage' in js,
            'Create new note': lambda js, code: 'add' in js.lower() or 'create' in js.lower(),
            'Delete note': lambda js, code: 'delete' in js.lower() or 'remove' in js.lower() or 'splice' in js,
            'XSS safe': lambda js, code: '&lt;' in js or '&amp;' in js or 'esc' in js.lower() or 'sanitize' in js.lower(),
        }
    },
]

async def main():
    for build in builds:
        short = build['desc'].split(' with ')[0] if ' with ' in build['desc'] else build['desc'][:35]

        # Show injected references
        refs = orig_select(build['desc'], build['scaffold'], max_refs=7)
        ref_labels = re.findall(r'### (.+)', refs)

        print(f'\n{"="*70}')
        print(f'  {short}')
        print(f'  References: {", ".join(r.split("(")[0].strip() for r in ref_labels[:4])}...')
        print(f'{"="*70}')

        tool = ApplicationBuilderTool(S(), llm, lambda: {'app_builder_max_tokens': 8192})

        async def progress(data):
            delta = data.get('_content_delta', '')
            if delta.strip():
                try: print(delta.rstrip())
                except UnicodeEncodeError: pass

        t0 = time.time()
        result = await tool.execute(
            description=build['desc'], scaffold=build['scaffold'],
            _progress_callback=progress, _request_model=MODEL)
        elapsed = time.time() - t0

        if not result.success:
            print(f'  FAILED ({elapsed:.0f}s): {result.error}')
            continue

        project = result.metadata['project']
        files = project.get('files', [])
        all_js = '\n'.join(f['content'] for f in files if f.get('role') in ('script', 'module'))
        all_code = '\n'.join(f['content'] for f in files)
        total_lines = all_code.count('\n') + 1

        print(f'\n  Result: {len(files)} files, {total_lines} lines, {elapsed:.0f}s, Score: {project.get("score", 0)}')
        for f in files:
            print(f'    {f["path"]:25s} ({f["role"]:6s}) {f["content"].count(chr(10))+1:4d} lines')

        passed = 0
        total = len(build['checks'])
        print('\n  Feature checks:')
        for name, check_fn in build['checks'].items():
            ok = check_fn(all_js, all_code)
            if ok: passed += 1
            print(f'    {"YES" if ok else "NO ":3s} | {name}')

        print(f'\n  Score: {passed}/{total} ({passed/total*100:.0f}%)')

asyncio.run(main())

"""Stress test Gemma 4B with 3 advanced builds."""
import asyncio, json, sys, re, time, urllib.request
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
from augmentum.tools.artifact_application import ApplicationBuilderTool

MODEL = 'gemma-3-4b-it'

async def llm(messages, max_tokens=4096, model='', **kw):
    payload = json.dumps({'model': MODEL, 'messages': messages, 'max_tokens': max(max_tokens, 4096), 'stream': False}).encode()
    req = urllib.request.Request('http://127.0.0.1:1234/v1/chat/completions', data=payload, headers={'Content-Type': 'application/json'})
    resp = urllib.request.urlopen(req, timeout=300)
    return json.loads(resp.read())['choices'][0]['message'].get('content', '')

class S:
    async def save(self, **kw): return {'id': 'x'}

builds = [
    {
        "desc": "kanban project board with 3 columns (To Do, In Progress, Done), drag and drop cards between columns, add new cards with title and priority color, delete cards, localStorage persistence, card count per column",
        "scaffold": "form",
        "checks": {
            'Has 3 columns': lambda js, html, css: 'todo' in html.lower() or 'to do' in html.lower() or 'to-do' in html.lower(),
            'Drag and drop': lambda js, html, css: 'dragstart' in js or 'draggable' in html,
            'Add cards': lambda js, html, css: 'add' in js.lower() and ('push' in js or 'append' in js.lower()),
            'Delete cards': lambda js, html, css: 'delete' in js.lower() or 'remove' in js.lower() or 'splice' in js,
            'localStorage': lambda js, html, css: 'localStorage' in js,
            'Priority/color': lambda js, html, css: 'priority' in js.lower() or 'color' in js.lower(),
            'XSS escaping': lambda js, html, css: '&lt;' in js or '&amp;' in js or 'esc(' in js or 'escape' in js.lower(),
            'CSS custom props': lambda js, html, css: '--' in css and ':root' in css,
            'Responsive': lambda js, html, css: '@media' in css,
            'IIFE': lambda js, html, css: '(function' in js,
        }
    },
    {
        "desc": "real-time cryptocurrency dashboard with 3 live-updating price charts (Bitcoin, Ethereum, Dogecoin), sparkline mini-charts in the sidebar, portfolio value calculator, dark theme with green/red price indicators, auto-refresh every 5 seconds with simulated data",
        "scaffold": "dashboard",
        "checks": {
            'Chart.js usage': lambda js, html, css: 'new Chart' in js or 'Chart(' in js,
            'Multiple charts': lambda js, html, css: js.count('Chart(') >= 2 or js.count('new Chart') >= 2,
            'Destroy before create': lambda js, html, css: '.destroy()' in js,
            'setInterval/refresh': lambda js, html, css: 'setInterval' in js or 'setTimeout' in js,
            'Green/red indicators': lambda js, html, css: ('green' in css.lower() or '#22c55e' in css or '#10b981' in css) and ('red' in css.lower() or '#ef4444' in css or '#f44' in css),
            'Dark theme': lambda js, html, css: any(c in css for c in ['#0c1', '#0a0', '#0f1', '#1a1', '#111', '#000', '#0d0', '#151', 'dark' ]),
            'Portfolio calc': lambda js, html, css: 'portfolio' in js.lower() or 'total' in js.lower() or 'value' in js.lower(),
            'CSS custom props': lambda js, html, css: '--' in css and ':root' in css,
            'Canvas elements': lambda js, html, css: '<canvas' in html,
            'IIFE': lambda js, html, css: '(function' in js,
        }
    },
    {
        "desc": "space invaders arcade game with player ship (arrow keys + spacebar), 5 rows of alien enemies that move side to side and descend, player bullets destroy aliens, alien bullets rain down randomly, score counter, 3 lives with hearts display, increasing difficulty each wave, game over and restart, retro pixel art style",
        "scaffold": "game",
        "checks": {
            'Canvas + getContext': lambda js, html, css: 'getContext' in js,
            'requestAnimationFrame': lambda js, html, css: 'requestAnimationFrame' in js,
            'Player input (arrows)': lambda js, html, css: 'ArrowLeft' in js or 'ArrowRight' in js or 'keydown' in js,
            'Shooting (spacebar)': lambda js, html, css: 'Space' in js or 'spacebar' in js.lower() or 'shoot' in js.lower(),
            'Enemy grid/rows': lambda js, html, css: 'row' in js.lower() or 'grid' in js.lower() or 'enemies' in js.lower() or 'alien' in js.lower(),
            'Collision detection': lambda js, html, css: 'collision' in js.lower() or 'hits' in js.lower() or 'intersect' in js.lower() or ('x + ' in js and 'width' in js),
            'Score tracking': lambda js, html, css: 'score' in js.lower(),
            'Lives/hearts': lambda js, html, css: 'lives' in js.lower() or 'heart' in js.lower() or 'life' in js.lower(),
            'Game states': lambda js, html, css: ('menu' in js.lower() or 'start' in js.lower()) and ('gameover' in js.lower() or 'game_over' in js.lower() or 'game over' in js.lower()),
            'IIFE or strict': lambda js, html, css: '(function' in js or 'use strict' in js,
        }
    },
]

async def main():
    for build in builds:
        short = build['desc'][:50]
        print(f'\n{"="*70}')
        print(f'  GEMMA 4B: {short}...')
        print(f'  Scaffold: {build["scaffold"]}')
        print(f'{"="*70}')

        tool = ApplicationBuilderTool(S(), llm, lambda: {'app_builder_max_tokens': 8192})
        t0 = time.time()

        async def progress(data):
            delta = data.get('_content_delta', '')
            if delta.strip():
                try: print(delta.rstrip())
                except UnicodeEncodeError: pass

        result = await tool.execute(
            description=build['desc'], scaffold=build['scaffold'],
            _progress_callback=progress, _request_model=MODEL)
        elapsed = time.time() - t0

        if not result.success:
            print(f'\n  FAILED ({elapsed:.0f}s): {result.error}')
            continue

        project = result.metadata['project']
        files = project.get('files', [])
        all_js = '\n'.join(f['content'] for f in files if f.get('role') in ('script', 'module'))
        all_html = '\n'.join(f['content'] for f in files if f.get('role') == 'entry')
        all_css = '\n'.join(f['content'] for f in files if f.get('role') == 'style')
        total_lines = sum(f['content'].count('\n') + 1 for f in files)

        print(f'\n  Result: {len(files)} files, {total_lines} lines, {elapsed:.0f}s, Score: {project.get("score", 0)}')
        for f in files:
            print(f'    {f["path"]:25s} ({f["role"]:6s}) {f["content"].count(chr(10))+1:4d} lines')

        # Quality checks
        css_vars = len(set(re.findall(r'--[\w-]+', all_css)))
        var_refs = all_css.count('var(')
        has_root = ':root' in all_css
        has_responsive = '@media' in all_css

        print(f'\n  CSS: {css_vars} custom props, {var_refs} var() refs, :root={has_root}, responsive={has_responsive}')
        print(f'  A11y: lang={"YES" if "lang=" in all_html else "NO"}, semantic={"YES" if any(t in all_html for t in ("<nav","<header","<footer","<main")) else "NO"}')

        # Feature checks
        passed = 0
        total = len(build['checks'])
        print(f'\n  Feature checks:')
        for name, check_fn in build['checks'].items():
            ok = check_fn(all_js, all_html, all_css)
            if ok: passed += 1
            print(f'    {"YES" if ok else "NO ":3s} | {name}')

        print(f'\n  SCORE: {passed}/{total} ({passed/total*100:.0f}%)')

asyncio.run(main())

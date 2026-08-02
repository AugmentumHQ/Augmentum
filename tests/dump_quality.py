"""Dump generated code with quality analysis."""
import asyncio, json, sys, re, urllib.request
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
from augmentum.tools.artifact_application import ApplicationBuilderTool

async def build(model, desc, scaffold):
    async def llm(messages, max_tokens=4096, **kw):
        payload = json.dumps({'model': model, 'messages': messages, 'max_tokens': max(max_tokens, 4096), 'stream': False}).encode()
        req = urllib.request.Request('http://127.0.0.1:1234/v1/chat/completions', data=payload, headers={'Content-Type': 'application/json'})
        return json.loads(urllib.request.urlopen(req, timeout=300).read())['choices'][0]['message'].get('content', '')

    class S:
        async def save(self, **kw): return {'id': 'x'}

    tool = ApplicationBuilderTool(S(), llm, lambda: {'app_builder_max_tokens': 8192})
    return await tool.execute(description=desc, scaffold=scaffold, _request_model=model)

async def main():
    model = sys.argv[1] if len(sys.argv) > 1 else 'gemma-3-4b-it'
    desc = sys.argv[2] if len(sys.argv) > 2 else 'simple calculator with add subtract multiply divide'
    scaffold = sys.argv[3] if len(sys.argv) > 3 else 'form'

    result = await build(model, desc, scaffold)
    if not result.success:
        print(f'FAILED: {result.error}')
        return

    files = result.metadata['project']['files']
    short = model.split('/')[-1][:20]
    print(f'\n{"#"*70}')
    print(f'  {short} | {desc[:45]} | {len(files)} files')
    print(f'{"#"*70}')

    for f in files:
        lines = f['content'].split('\n')
        max_show = 35 if f['role'] == 'entry' else 50 if f['role'] == 'style' else 60
        print(f'\n--- {f["path"]} ({f["role"]}, {len(lines)} lines) ---')
        for i, line in enumerate(lines[:max_show]):
            print(f'{i+1:3d} | {line}')
        if len(lines) > max_show:
            print(f'    ... ({len(lines) - max_show} more lines)')

    # Quality analysis
    all_code = '\n'.join(f['content'] for f in files)
    all_css = '\n'.join(f['content'] for f in files if f['role'] == 'style')
    all_js = '\n'.join(f['content'] for f in files if f['role'] in ('script', 'module'))
    entry = next((f['content'] for f in files if f['role'] == 'entry'), '')

    css_vars = set(re.findall(r'--[\w-]+', all_css))
    var_refs = all_css.count('var(')

    print(f'\n{"="*50}')
    print(f'  QUALITY REPORT')
    print(f'{"="*50}')
    print(f'  Files: {len(files)}, Total lines: {sum(f["content"].count(chr(10))+1 for f in files)}')
    print()
    print(f'  ACCESSIBILITY')
    print(f'    lang="en":        {"YES" if "lang=" in entry else "NO"}')
    print(f'    alt on images:    {"YES" if "<img" not in entry or "alt=" in entry else "NO"}')
    print(f'    aria-label count: {all_code.count("aria-label")}')
    print(f'    <main> landmark:  {"YES" if "<main" in entry else "NO"}')
    print(f'    Semantic tags:    {"YES" if any(t in entry for t in ("<nav","<header","<footer","<main","<section","<article")) else "NO"}')
    print()
    print(f'  CSS QUALITY')
    print(f'    Custom properties: {len(css_vars)}')
    print(f'    var() references:  {var_refs}')
    print(f'    Has :root block:   {"YES" if ":root" in all_css else "NO"}')
    print(f'    @media responsive: {"YES" if "@media" in all_css else "NO"}')
    print(f'    Transitions:       {all_css.count("transition")}')
    print(f'    Hover states:      {all_css.count(":hover")}')
    print()
    print(f'  JS QUALITY')
    print(f'    IIFE:              {"YES" if "(function()" in all_js or "(function(" in all_js else "NO"}')
    print(f'    use strict:        {"YES" if "use strict" in all_js else "NO"}')
    print(f'    XSS escaping:      {"YES" if "esc(" in all_js or "&lt;" in all_js or "escapeHtml" in all_js else "NO"}')
    print(f'    preventDefault:    {"YES" if "preventDefault" in all_js else "NO"}')
    print(f'    No alert():        {"YES" if "alert(" not in all_js else "NO"}')
    print(f'    No eval():         {"YES" if "eval(" not in all_js else "NO"}')
    print(f'    localStorage:      {"YES" if "localStorage" in all_js else "NO"}')
    print(f'    DOMContentLoaded:  {"YES" if "DOMContentLoaded" in all_js else "NO"}')

asyncio.run(main())

"""Reference implementation library for the application builder.

Each reference is a tagged, compact code skeleton that demonstrates correct
patterns for a specific type of application.  The builder dynamically selects
the 2-3 most relevant references based on the user's description, scaffold
type, and keyword matching.

Why this works: models are far better at pattern-matching against working code
examples than following text instructions.  A 4B model that ignores "use
requestAnimationFrame with delta-time" will reliably copy the pattern when it
sees a working game loop in context.

Token budget: each reference is ~100-200 tokens.  2-3 selected per build
= ~300-500 tokens, well under 6% of the generation budget.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Reference Library
# ---------------------------------------------------------------------------
# Each entry: scaffold(s) it applies to, keywords for matching, the code.
# The "always" flag means it's included for that scaffold regardless of keywords.

REFERENCES: list[dict] = [
    # ===== STATIC =====
    {
        "id": "static-base",
        "scaffolds": ["static", "form", "dashboard"],
        "keywords": [],
        "always_for": ["static"],
        "label": "Base web app (polished design system + semantic HTML)",
        "code": """\
/* styles.css — Design system foundation. EVERY app should start from this. */
:root {
  /* Color palette — cohesive, accessible contrast ratios */
  --bg: #fafafa; --bg-card: #ffffff; --bg-hover: #f1f5f9;
  --text: #0f172a; --text-dim: #64748b; --text-muted: #94a3b8;
  --accent: #6366f1; --accent-hover: #4f46e5; --accent-glow: rgba(99,102,241,0.15);
  --success: #10b981; --error: #ef4444; --warning: #f59e0b;
  --border: #e2e8f0; --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
  --shadow-lg: 0 10px 25px rgba(0,0,0,0.07), 0 4px 10px rgba(0,0,0,0.04);
  /* Typography scale */
  --font: 'Inter', system-ui, -apple-system, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
  --text-xs: 0.75rem; --text-sm: 0.875rem; --text-base: 1rem; --text-lg: 1.125rem; --text-xl: 1.25rem; --text-2xl: 1.5rem; --text-3xl: 2rem;
  /* Spacing scale (consistent rhythm) */
  --space-xs: 0.25rem; --space-sm: 0.5rem; --space-md: 1rem; --space-lg: 1.5rem; --space-xl: 2rem; --space-2xl: 3rem;
  --radius-sm: 6px; --radius-md: 10px; --radius-lg: 16px; --radius-full: 9999px;
  --transition: 150ms cubic-bezier(0.4, 0, 0.2, 1);
}
* { box-sizing: border-box; margin: 0; }
body { font-family: var(--font); background: var(--bg); color: var(--text); line-height: 1.6;
  -webkit-font-smoothing: antialiased; }
/* Buttons — always have hover, active, focus-visible states */
button, .btn { display: inline-flex; align-items: center; gap: var(--space-sm); padding: var(--space-sm) var(--space-lg);
  border: none; border-radius: var(--radius-md); font-weight: 600; font-size: var(--text-sm);
  cursor: pointer; transition: all var(--transition); }
.btn-primary { background: var(--accent); color: white; }
.btn-primary:hover { background: var(--accent-hover); box-shadow: 0 0 0 3px var(--accent-glow); transform: translateY(-1px); }
.btn-primary:active { transform: translateY(0); }
.btn-secondary { background: transparent; color: var(--text); border: 1px solid var(--border); }
.btn-secondary:hover { background: var(--bg-hover); border-color: var(--accent); }
/* Focus rings — visible on keyboard, hidden on mouse (accessibility) */
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
:focus:not(:focus-visible) { outline: none; }
/* Cards */
.card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-lg);
  padding: var(--space-lg); box-shadow: var(--shadow); transition: all var(--transition); }
.card:hover { box-shadow: var(--shadow-lg); border-color: var(--accent-glow); }
/* Inputs */
input, select, textarea { padding: var(--space-sm) var(--space-md); border: 1px solid var(--border);
  border-radius: var(--radius-md); font-size: var(--text-sm); font-family: inherit; background: var(--bg-card);
  color: var(--text); transition: border var(--transition), box-shadow var(--transition); width: 100%; }
input:focus, select:focus, textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow); }
/* Responsive */
@media (max-width: 640px) { :root { --text-3xl: 1.5rem; --text-2xl: 1.25rem; --space-xl: 1.5rem; } }

// app.js — IIFE + DOMContentLoaded + strict mode
(function() {
  'use strict';
  document.addEventListener('DOMContentLoaded', () => {
    const app = document.getElementById('app');
  });
})();""",
    },
    {
        "id": "static-interactive",
        "scaffolds": ["static"],
        "keywords": ["interactive", "utility", "tool", "converter", "generator", "search", "filter", "list"],
        "always_for": [],
        "label": "Interactive utility (state + render + filter)",
        "code": """\
// State-driven rendering pattern
(function() {
  'use strict';
  const state = { items: [], filter: '' };
  function esc(s) { return s.replace(/[<>&"]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'})[c]); }
  function render() {
    const filtered = state.items.filter(i => !state.filter || i.includes(state.filter));
    document.getElementById('app').innerHTML = filtered.map(i => '<div class="item">' + esc(i) + '</div>').join('');
    document.getElementById('count').textContent = filtered.length;
  }
  document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('search').addEventListener('input', (e) => { state.filter = e.target.value; render(); });
    render();
  });
})();""",
    },
    {
        "id": "static-shop",
        "scaffolds": ["static", "form"],
        "keywords": ["shop", "store", "product", "cart", "ecommerce", "buy", "price", "checkout", "catalog"],
        "always_for": [],
        "label": "Shop/product listing (cart state + product grid)",
        "code": """\
// E-commerce pattern: product grid + cart state + render
(function() {
  'use strict';
  const cart = [];
  function esc(s) { return s.replace(/[<>&"]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'})[c]); }
  function addToCart(product) { cart.push(product); renderCart(); }
  function renderCart() {
    const el = document.getElementById('cart');
    el.innerHTML = cart.map(p => '<div class="cart-item">' + esc(p.name) + ' $' + p.price.toFixed(2) + '</div>').join('');
    document.getElementById('cart-total').textContent = '$' + cart.reduce((s, p) => s + p.price, 0).toFixed(2);
    document.getElementById('cart-count').textContent = cart.length;
  }
  function renderProducts(products) {
    document.getElementById('products').innerHTML = products.map(p =>
      '<div class="product-card"><h3>' + esc(p.name) + '</h3><span class="price">$' + p.price.toFixed(2) + '</span>' +
      '<button data-id="' + p.id + '">Add to Cart</button></div>'
    ).join('');
    document.getElementById('products').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-id]');
      if (btn) addToCart(products.find(p => p.id === btn.dataset.id));
    });
  }
  document.addEventListener('DOMContentLoaded', () => { renderProducts(window.PRODUCTS || []); });
})();""",
    },
    {
        "id": "static-landing",
        "scaffolds": ["static"],
        "keywords": ["landing", "portfolio", "hero", "showcase", "gallery", "sections", "scroll", "parallax"],
        "always_for": [],
        "label": "Landing page (scroll sections + intersection observer)",
        "code": """\
// Landing page pattern: scroll-triggered animations
(function() {
  'use strict';
  document.addEventListener('DOMContentLoaded', () => {
    // Reveal sections on scroll
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('visible'); observer.unobserve(e.target); } });
    }, { threshold: 0.15 });
    document.querySelectorAll('.section').forEach(s => observer.observe(s));
    // Smooth scroll for nav links
    document.querySelectorAll('a[href^="#"]').forEach(a => {
      a.addEventListener('click', (e) => { e.preventDefault(); document.querySelector(a.getAttribute('href'))?.scrollIntoView({ behavior: 'smooth' }); });
    });
  });
})();
/* CSS: .section { opacity: 0; transform: translateY(20px); transition: all 0.6s; } .section.visible { opacity: 1; transform: none; } */""",
    },

    # ===== DASHBOARD =====
    {
        "id": "dashboard-base",
        "scaffolds": ["dashboard"],
        "keywords": [],
        "always_for": ["dashboard"],
        "label": "Chart.js dashboard (polished dark theme + grid + destroy-before-create)",
        "code": """\
/* styles.css — Dashboard dark theme with depth and glow accents */
:root {
  --bg: #0c1222; --bg-card: #151d30; --bg-card-hover: #1a2540;
  --text: #e2e8f0; --text-dim: #94a3b8; --text-muted: #475569;
  --accent: #818cf8; --accent-dim: rgba(129,140,248,0.12); --accent-glow: rgba(129,140,248,0.2);
  --success: #34d399; --error: #f87171; --warning: #fbbf24;
  --border: rgba(255,255,255,0.06);
  --shadow-card: 0 4px 20px rgba(0,0,0,0.25);
  --font: 'Inter', system-ui, sans-serif; --radius: 14px;
}
.dashboard { display: grid; grid-template-columns: 260px 1fr; min-height: 100vh; }
.sidebar { background: var(--bg-card); border-right: 1px solid var(--border); padding: 2rem; }
.sidebar .brand { font-size: 1.4rem; font-weight: 700; color: var(--accent); letter-spacing: -0.02em; }
.main { padding: 2rem; overflow-y: auto; }
/* KPI cards — glassmorphism subtle effect */
.stats-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.25rem; margin-bottom: 2rem; }
.stat-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 1.5rem; box-shadow: var(--shadow-card); transition: all 200ms ease; }
.stat-card:hover { border-color: var(--accent-dim); transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); }
.stat-value { font-size: 2rem; font-weight: 700; color: var(--accent); margin-top: 0.25rem;
  background: linear-gradient(135deg, var(--accent), #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.stat-label { font-size: 0.8rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }
/* Chart cards */
.charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1.25rem; }
.chart-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 1.5rem; box-shadow: var(--shadow-card); }
.chart-title { font-size: 1rem; font-weight: 600; color: var(--text); margin-bottom: 1rem;
  padding-bottom: 0.75rem; border-bottom: 1px solid var(--border); }
/* Form controls */
select, button { background: var(--bg-card-hover); border: 1px solid var(--border); color: var(--text);
  padding: 0.6rem 1rem; border-radius: 8px; font-size: 0.85rem; transition: all 150ms; }
button:hover { background: var(--accent); color: white; border-color: var(--accent); }
@media (max-width: 768px) { .dashboard { grid-template-columns: 1fr; } .sidebar { display: none; }
  .charts-grid { grid-template-columns: 1fr; } }

// app.js — Chart.js with consistent theming
(function() {
  'use strict';
  let chartInstance = null;
  // Chart.js global defaults — match our CSS theme
  Chart.defaults.color = '#94a3b8';
  Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
  Chart.defaults.font.family = 'Inter, system-ui, sans-serif';
  function initChart(canvasId, config) {
    if (chartInstance) chartInstance.destroy(); // CRITICAL: prevents canvas overlay
    chartInstance = new Chart(document.getElementById(canvasId).getContext('2d'), config);
  }
  document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('filter-form')?.addEventListener('submit', (e) => { e.preventDefault(); /* reinit */ });
  });
})();""",
    },
    {
        "id": "dashboard-multi",
        "scaffolds": ["dashboard"],
        "keywords": ["multiple", "charts", "analytics", "sections", "kpi", "metrics"],
        "always_for": [],
        "label": "Multi-chart with centralized data",
        "code": """\
// Multiple charts: track by ID, centralized data source
(function() {
  'use strict';
  const charts = {};
  function createOrUpdate(canvasId, config) {
    if (charts[canvasId]) charts[canvasId].destroy();
    charts[canvasId] = new Chart(document.getElementById(canvasId).getContext('2d'), config);
  }
  function loadDashboard(data) {
    createOrUpdate('chart-revenue', { type: 'line', data: { labels: data.months, datasets: [{ data: data.revenue }] } });
    createOrUpdate('chart-sales', { type: 'bar', data: { labels: data.categories, datasets: [{ data: data.sales }] } });
    document.getElementById('kpi-total').textContent = '$' + data.revenue.reduce((a,b) => a+b, 0).toLocaleString();
  }
  document.addEventListener('DOMContentLoaded', () => loadDashboard(generateData()));
})();""",
    },
    {
        "id": "dashboard-realtime",
        "scaffolds": ["dashboard"],
        "keywords": ["live", "realtime", "real-time", "monitor", "polling", "streaming", "update", "refresh"],
        "always_for": [],
        "label": "Real-time updating (setInterval + rolling window)",
        "code": """\
// Live dashboard: rolling data window with periodic updates
(function() {
  'use strict';
  let chart = null;
  const MAX_POINTS = 20, dataPoints = [];
  function addPoint(v) { dataPoints.push(v); if (dataPoints.length > MAX_POINTS) dataPoints.shift(); }
  function updateChart() {
    if (chart) chart.destroy();
    chart = new Chart(document.getElementById('live-chart').getContext('2d'), {
      type: 'line', data: { labels: dataPoints.map((_, i) => i), datasets: [{ data: [...dataPoints], borderColor: '#22c55e', tension: 0.3 }] },
      options: { responsive: true, animation: { duration: 200 }, scales: { y: { min: 0 } } }
    });
  }
  document.addEventListener('DOMContentLoaded', () => { setInterval(() => { addPoint(Math.random() * 100); updateChart(); }, 2000); });
})();""",
    },

    # ===== GAME =====
    {
        "id": "game-base",
        "scaffolds": ["game"],
        "keywords": [],
        "always_for": ["game"],
        "label": "Game loop + Entity + neon cyberpunk aesthetic (core pattern for ALL games)",
        "code": """\
/* Neon Cyberpunk Game — dark bg, glow effects, scanline overlay, arcade feel */
:root{--bg:#0a0a1a;--neon:#0ff;--pink:#f0f;--glow:0 0 10px var(--neon),0 0 40px var(--neon)}
body{background:var(--bg);overflow:hidden;font-family:'Courier New',monospace}
canvas{display:block}
#hud{position:fixed;top:12px;left:16px;color:var(--neon);font-size:14px;text-shadow:var(--glow);z-index:2;letter-spacing:2px}
#overlay{position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:3;background:rgba(10,10,26,.85)}
#overlay h1{font-size:4rem;color:var(--neon);text-shadow:0 0 20px var(--neon),0 0 60px var(--pink);animation:pulse 1.5s ease-in-out infinite}
#overlay p{color:var(--pink);margin-top:1rem;font-size:1.1rem;text-shadow:0 0 8px var(--pink);letter-spacing:3px}
.gameover h1{animation:glitch .3s infinite!important;color:var(--pink)!important}
#scanlines{position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,255,255,.03) 2px,rgba(0,255,255,.03) 4px);pointer-events:none;z-index:10}
@keyframes pulse{0%,100%{opacity:1;filter:brightness(1)}50%{opacity:.7;filter:brightness(1.5)}}
@keyframes glitch{0%{clip-path:inset(40% 0 20% 0)}25%{clip-path:inset(10% 0 60% 0)}50%{clip-path:inset(70% 0 5% 0)}75%{clip-path:inset(20% 0 40% 0)}100%{clip-path:inset(50% 0 10% 0)}}

// game.js — IIFE with Entity, game loop, input, state machine
(function(){const C=document.getElementById('game'),X=C.getContext('2d'),hud=document.getElementById('hud'),ov=document.getElementById('overlay');
const resize=()=>{C.width=innerWidth;C.height=innerHeight};addEventListener('resize',resize);resize();
const keys={};addEventListener('keydown',e=>{keys[e.code]=true;e.preventDefault()});addEventListener('keyup',e=>{keys[e.code]=false});
class Entity{constructor(x,y,w,h,col){Object.assign(this,{x,y,w,h,col,vx:0,vy:0,dead:false})}
update(dt){this.x+=this.vx*dt;this.y+=this.vy*dt}
draw(ctx){ctx.shadowColor=this.col;ctx.shadowBlur=15;ctx.fillStyle=this.col;ctx.fillRect(this.x,this.y,this.w,this.h);ctx.shadowBlur=0}
hits(o){return this.x<o.x+o.w&&this.x+this.w>o.x&&this.y<o.y+o.h&&this.y+this.h>o.y}}
let state='menu',score=0,entities=[];
function startGame(){state='playing';score=0;entities=[];ov.style.display='none';last=performance.now();requestAnimationFrame(loop)}
function endGame(){state='gameover';ov.style.display='flex';ov.classList.add('gameover');ov.querySelector('h1').textContent='GAME OVER'}
let last=0;function loop(t){const dt=Math.min((t-last)/1000,.05);last=t;
if(state!=='playing')return;
X.fillStyle='#0a0a1a';X.fillRect(0,0,C.width,C.height);
for(let i=entities.length-1;i>=0;i--){entities[i].update(dt);if(entities[i].dead)entities.splice(i,1)}
entities.forEach(e=>e.draw(X));
hud.textContent='SCORE '+String(score).padStart(6,'0');score+=Math.round(dt*10);
requestAnimationFrame(loop)}
addEventListener('keydown',e=>{if(e.key==='Enter'&&state!=='playing')startGame()})})();""",
    },
    {
        "id": "game-shooter",
        "scaffolds": ["game"],
        "keywords": ["shoot", "shooter", "bullet", "enemy", "wave", "space", "invader", "asteroid"],
        "always_for": [],
        "label": "Shooter (bullets + enemy waves + spawning)",
        "code": """\
// Shooter extension: bullets, enemy spawning, wave progression
class Bullet extends Entity {
  constructor(x, y, dy) { super(x, y, 4, 12); this.dy = dy; }
  update(dt) { this.y += this.dy * dt * 0.01; if (this.y < 0 || this.y > canvas.height) this.dead = true; }
  draw(ctx) { ctx.fillStyle = '#ff0'; super.draw(ctx); }
}
class Enemy extends Entity {
  constructor(x, y, speed) { super(x, y, 32, 32); this.speed = speed; }
  update(dt) { this.y += this.speed * dt * 0.01; if (this.y > canvas.height) this.dead = true; }
  draw(ctx) { ctx.fillStyle = '#f44'; super.draw(ctx); }
}
// Spawn pattern: timer-based with increasing difficulty
let spawnTimer = 0;
function spawnWave(dt) {
  spawnTimer -= dt;
  if (spawnTimer <= 0) { state.entities.push(new Enemy(Math.random() * canvas.width, -40, 100 + state.score * 2)); spawnTimer = 1500; }
}""",
    },
    {
        "id": "game-platformer",
        "scaffolds": ["game"],
        "keywords": ["platform", "platformer", "jump", "gravity", "side-scroll", "run", "mario", "climb"],
        "always_for": [],
        "label": "Platformer (gravity + jumping + ground collision)",
        "code": """\
// Platformer extension: gravity, jumping, ground check
const GRAVITY = 0.5, JUMP = -12;
class Player extends Entity {
  constructor(x, y) { super(x, y, 32, 48); this.vy = 0; this.grounded = false; }
  update(dt) {
    if (keys['ArrowLeft']) this.x -= 4; if (keys['ArrowRight']) this.x += 4;
    if (keys['Space'] && this.grounded) { this.vy = JUMP; this.grounded = false; }
    this.vy += GRAVITY; this.y += this.vy;
    // Ground collision (replace with platform collision for real game)
    if (this.y > canvas.height - 60 - this.h) { this.y = canvas.height - 60 - this.h; this.vy = 0; this.grounded = true; }
    this.x = Math.max(0, Math.min(canvas.width - this.w, this.x));
  }
  draw(ctx) { ctx.fillStyle = '#0ff'; super.draw(ctx); }
}""",
    },
    {
        "id": "game-puzzle",
        "scaffolds": ["game"],
        "keywords": ["puzzle", "grid", "tile", "match", "card", "memory", "tetris", "block", "board"],
        "always_for": [],
        "label": "Grid/puzzle (tile grid + click interaction)",
        "code": """\
// Grid game: 2D array + click-to-tile mapping
const COLS = 8, ROWS = 8, TILE = 60;
const grid = Array.from({ length: ROWS }, () => Array.from({ length: COLS }, () => Math.floor(Math.random() * 4)));
function drawGrid() {
  const colors = ['#ff4444', '#44ff44', '#4444ff', '#ffff44'];
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
    ctx.fillStyle = colors[grid[r][c]]; ctx.fillRect(c * TILE + 2, r * TILE + 2, TILE - 4, TILE - 4);
  }
}
canvas.addEventListener('click', (e) => {
  const c = Math.floor(e.offsetX / TILE), r = Math.floor(e.offsetY / TILE);
  if (r >= 0 && r < ROWS && c >= 0 && c < COLS) { grid[r][c] = (grid[r][c] + 1) % 4; drawGrid(); }
});""",
    },
    {
        "id": "game-physics",
        "scaffolds": ["game"],
        "keywords": ["physics", "particle", "gravity", "bounce", "collision", "force", "velocity", "simulate", "pendulum", "orbit"],
        "always_for": [],
        "label": "Physics simulation (velocity + force + particle system)",
        "code": """\
// Physics pattern: velocity vectors, force accumulation, elastic collision
class Particle extends Entity {
  constructor(x, y, r) { super(x - r, y - r, r*2, r*2); this.r = r; this.vx = 0; this.vy = 0; this.mass = r; }
  applyForce(fx, fy) { this.vx += fx / this.mass; this.vy += fy / this.mass; }
  update(dt) {
    this.applyForce(0, 0.1); // gravity
    this.x += this.vx; this.y += this.vy;
    this.vx *= 0.999; this.vy *= 0.999; // friction
    // Wall bounce
    if (this.x < 0) { this.x = 0; this.vx *= -0.8; }
    if (this.x + this.w > canvas.width) { this.x = canvas.width - this.w; this.vx *= -0.8; }
    if (this.y + this.h > canvas.height) { this.y = canvas.height - this.h; this.vy *= -0.8; }
  }
  draw(ctx) { ctx.beginPath(); ctx.arc(this.x+this.r, this.y+this.r, this.r, 0, Math.PI*2); ctx.fillStyle='#0ff'; ctx.fill(); }
}
// Elastic collision between two circles
function collide(a, b) {
  const dx = (b.x+b.r)-(a.x+a.r), dy = (b.y+b.r)-(a.y+a.r), dist = Math.sqrt(dx*dx+dy*dy);
  if (dist < a.r + b.r) { const nx=dx/dist, ny=dy/dist, dvx=a.vx-b.vx, dvy=a.vy-b.vy, dot=dvx*nx+dvy*ny;
    a.vx -= dot*nx; a.vy -= dot*ny; b.vx += dot*nx; b.vy += dot*ny; }
}""",
    },
    {
        "id": "game-snake",
        "scaffolds": ["game"],
        "keywords": ["snake", "grow", "tail", "food", "classic", "nokia"],
        "always_for": [],
        "label": "Snake-type (grid movement + growing body + food)",
        "code": """\
// Snake pattern: direction-based grid movement, body array, food collision
const GRID = 20, SPEED = 150;
let snake = [{ x: 5, y: 5 }], dir = { x: 1, y: 0 }, food = { x: 10, y: 10 }, moveTimer = 0;
function placeFood() { food = { x: Math.floor(Math.random() * (canvas.width / GRID)), y: Math.floor(Math.random() * (canvas.height / GRID)) }; }
function updateSnake(dt) {
  moveTimer += dt; if (moveTimer < SPEED) return; moveTimer = 0;
  const head = { x: snake[0].x + dir.x, y: snake[0].y + dir.y };
  // Wall wrap or game over
  if (head.x < 0 || head.y < 0 || head.x >= canvas.width/GRID || head.y >= canvas.height/GRID) { state.phase = 'gameover'; return; }
  // Self collision
  if (snake.some(s => s.x === head.x && s.y === head.y)) { state.phase = 'gameover'; return; }
  snake.unshift(head);
  if (head.x === food.x && head.y === food.y) { state.score += 10; placeFood(); } else { snake.pop(); }
}
window.addEventListener('keydown', (e) => {
  if (e.code === 'ArrowUp' && dir.y === 0) dir = { x: 0, y: -1 };
  if (e.code === 'ArrowDown' && dir.y === 0) dir = { x: 0, y: 1 };
  if (e.code === 'ArrowLeft' && dir.x === 0) dir = { x: -1, y: 0 };
  if (e.code === 'ArrowRight' && dir.x === 0) dir = { x: 1, y: 0 };
});""",
    },

    # ===== FORM =====
    {
        "id": "form-base",
        "scaffolds": ["form"],
        "keywords": [],
        "always_for": ["form"],
        "label": "Form with warm organic design (floating labels + validation + persistence)",
        "code": """\
/* Warm organic design — soft pastels, generous spacing, spring transitions */
:root{--bg:linear-gradient(135deg,#fde1d3,#e8d5f5);--card:#fff9f7;--input-bg:#faf5f2;
  --border:#f0e0d8;--focus:#d4b8e8;--text:#5a4a5e;--label:#9b8a9e;
  --error:#e8836a;--success:#7ec8a0;--shadow:0 8px 32px rgba(180,140,170,.15);
  --font:'Avenir','Nunito',system-ui,sans-serif;--radius:20px;--spring:cubic-bezier(.34,1.56,.64,1)}
*{box-sizing:border-box;margin:0}
body{font-family:var(--font);background:var(--bg);min-height:100vh;display:grid;place-items:center;padding:2rem;color:var(--text)}
.card{background:var(--card);border-radius:calc(var(--radius)*1.5);padding:2.5rem;width:min(420px,100%);box-shadow:var(--shadow)}
/* Floating labels — animate up on focus/filled */
.field{position:relative;margin-bottom:1.5rem}
.field input{width:100%;padding:1.2rem 1rem .6rem;font:inherit;font-size:1rem;background:var(--input-bg);
  border:2px solid var(--border);border-radius:var(--radius);outline:none;transition:border .3s var(--spring),box-shadow .3s}
.field label{position:absolute;left:1rem;top:50%;transform:translateY(-50%);color:var(--label);pointer-events:none;
  transition:.25s var(--spring);font-size:.95rem}
.field input:focus,.field input:not(:placeholder-shown){border-color:var(--focus);box-shadow:0 4px 16px rgba(212,184,232,.25)}
.field input:focus+label,.field input:not(:placeholder-shown)+label{top:.55rem;transform:none;font-size:.7rem;color:var(--focus)}
.field .err{color:var(--error);font-size:.78rem;margin-top:.3rem;min-height:1.1rem}
button{width:100%;padding:1rem;font:inherit;font-size:1.05rem;font-weight:600;border:none;border-radius:var(--radius);
  background:linear-gradient(135deg,#f0b8a8,#d4b8e8);color:#fff;cursor:pointer;transition:.3s var(--spring);
  box-shadow:0 4px 16px rgba(212,184,232,.3)}
button:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(212,184,232,.4)}
#results div{padding:1rem;background:var(--input-bg);border-radius:calc(var(--radius)*.8);margin-top:.5rem;
  border-left:3px solid var(--success);animation:fadeIn .4s var(--spring)}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}

// app.js — form with validation, persistence, XSS escaping
(function(){'use strict';
const esc=s=>String(s).replace(/[&<>"'`]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;','`':'&#96;'})[c]).replace(/\\$\\{/g,'&#36;{');
const KEY='form_data_v1';
function load(){try{return JSON.parse(localStorage.getItem(KEY))||[]}catch{return[]}}
function save(d){localStorage.setItem(KEY,JSON.stringify(d))}
function validate(el,test,msg){const e=el.parentElement.querySelector('.err');
  if(!test){e.textContent=msg;el.style.borderColor='var(--error)';return false}e.textContent='';el.style.borderColor='';return true}
function render(entries){document.getElementById('results').innerHTML=entries.map(e=>
  '<div><strong>'+esc(e.name)+'</strong></div>').join('')}
render(load());
document.getElementById('main-form').addEventListener('submit',function(e){e.preventDefault();
  const n=document.getElementById('name');
  if(!validate(n,n.value.trim().length>=2,'Name must be at least 2 characters'))return;
  const entries=load();entries.push({name:n.value.trim(),ts:Date.now()});save(entries);render(entries);this.reset()})
})();""",
    },
    {
        "id": "form-wizard",
        "scaffolds": ["form"],
        "keywords": ["wizard", "multi-step", "steps", "progress", "next", "back", "onboarding", "survey"],
        "always_for": [],
        "label": "Multi-step wizard (step navigation + per-step validation)",
        "code": """\
// Multi-step wizard pattern
(function() {
  'use strict';
  const steps = document.querySelectorAll('.wizard-step');
  let current = 0;
  function showStep(i) { steps.forEach((s, j) => s.classList.toggle('hidden', j !== i));
    document.getElementById('btn-prev').disabled = i === 0;
    document.getElementById('btn-next').textContent = i === steps.length - 1 ? 'Submit' : 'Next';
  }
  function validateStep(i) { let ok = true; steps[i].querySelectorAll('[required]').forEach(inp => {
    if (!inp.value.trim()) { document.getElementById('err-' + inp.id).textContent = 'Required'; ok = false; }
    else { document.getElementById('err-' + inp.id).textContent = ''; } }); return ok; }
  document.getElementById('btn-next').addEventListener('click', () => { if (!validateStep(current)) return;
    if (current < steps.length - 1) { current++; showStep(current); } else { /* submit */ } });
  document.getElementById('btn-prev').addEventListener('click', () => { if (current > 0) { current--; showStep(current); } });
  showStep(0);
})();""",
    },
    {
        "id": "form-crud",
        "scaffolds": ["form", "static"],
        "keywords": ["crud", "create", "delete", "edit", "manage", "list", "todo", "task", "items", "board", "kanban"],
        "always_for": [],
        "label": "CRUD data manager (list + create + delete + persist)",
        "code": """\
// CRUD pattern: localStorage + event delegation + render cycle
(function() {
  'use strict';
  const KEY = 'items_v1';
  function load() { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch { return []; } }
  function save(items) { localStorage.setItem(KEY, JSON.stringify(items)); }
  function esc(s) { return s.replace(/[<>&"]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'})[c]); }
  let items = load();
  function render() {
    document.getElementById('item-list').innerHTML = items.map((item, i) =>
      '<div class="item"><span>' + esc(item.title) + '</span><button data-del="' + i + '">Delete</button></div>').join('');
  }
  document.getElementById('item-list').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-del]');
    if (btn) { items.splice(+btn.dataset.del, 1); save(items); render(); }
  });
  document.getElementById('add-form').addEventListener('submit', (e) => {
    e.preventDefault(); const inp = document.getElementById('new-title');
    if (!inp.value.trim()) return; items.push({ title: inp.value.trim(), created: Date.now() });
    save(items); render(); inp.value = '';
  });
  render();
})();""",
    },
    {
        "id": "form-calculator",
        "scaffolds": ["form"],
        "keywords": ["calculator", "calc", "compute", "convert", "converter", "unit", "math", "bmi", "tip"],
        "always_for": [],
        "label": "Calculator/converter (input → compute → display result)",
        "code": """\
// Calculator pattern: inputs → compute → show result (no alert)
(function() {
  'use strict';
  function compute() {
    const a = parseFloat(document.getElementById('input-a').value) || 0;
    const b = parseFloat(document.getElementById('input-b').value) || 0;
    const op = document.getElementById('operation').value;
    let result;
    switch (op) { case '+': result = a + b; break; case '-': result = a - b; break;
      case '*': result = a * b; break; case '/': result = b !== 0 ? a / b : 'Error: div/0'; break; }
    document.getElementById('result').textContent = typeof result === 'number' ? result.toLocaleString() : result;
    document.getElementById('result-section').classList.remove('hidden');
  }
  document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('calc-form').addEventListener('submit', (e) => { e.preventDefault(); compute(); });
    // Live calculation on input change (optional)
    document.querySelectorAll('#input-a, #input-b, #operation').forEach(el => el.addEventListener('input', compute));
  });
})();""",
    },

    # ===== GAME — Additional Types =====
    {
        "id": "game-racing",
        "scaffolds": ["game"],
        "keywords": ["race", "racing", "car", "drive", "track", "speed", "drift", "road"],
        "always_for": [],
        "label": "Racing/driving (top-down movement + track)",
        "code": """\
// Racing: rotation-based movement, speed/acceleration
class Car extends Entity {
  constructor(x, y) { super(x, y, 20, 40); this.angle = -Math.PI/2; this.speed = 0; this.maxSpeed = 5; }
  update(dt) {
    if (keys['ArrowUp']) this.speed = Math.min(this.maxSpeed, this.speed + 0.1);
    if (keys['ArrowDown']) this.speed = Math.max(-2, this.speed - 0.1);
    if (keys['ArrowLeft']) this.angle -= 0.04 * (this.speed / this.maxSpeed);
    if (keys['ArrowRight']) this.angle += 0.04 * (this.speed / this.maxSpeed);
    this.speed *= 0.98; // friction
    this.x += Math.cos(this.angle) * this.speed; this.y += Math.sin(this.angle) * this.speed;
  }
  draw(ctx) { ctx.save(); ctx.translate(this.x, this.y); ctx.rotate(this.angle);
    ctx.fillStyle = '#0af'; ctx.fillRect(-10, -20, 20, 40); ctx.restore(); }
}""",
    },
    {
        "id": "game-tower-defense",
        "scaffolds": ["game"],
        "keywords": ["tower", "defense", "path", "wave", "spawn", "defend", "strategy", "place", "turret"],
        "always_for": [],
        "label": "Tower defense (path following + placement)",
        "code": """\
// Tower defense: enemies follow path, towers placed on grid
const PATH = [{x:0,y:200},{x:200,y:200},{x:200,y:400},{x:600,y:400}];
class PathEnemy extends Entity {
  constructor() { super(PATH[0].x, PATH[0].y, 16, 16); this.pathIdx = 0; this.hp = 3; this.speed = 1; }
  update(dt) {
    const target = PATH[this.pathIdx + 1]; if (!target) { this.dead = true; state.lives--; return; }
    const dx = target.x - this.x, dy = target.y - this.y, dist = Math.sqrt(dx*dx+dy*dy);
    if (dist < 4) { this.pathIdx++; } else { this.x += (dx/dist)*this.speed; this.y += (dy/dist)*this.speed; }
    if (this.hp <= 0) { this.dead = true; state.score += 10; }
  }
}
class Tower { constructor(x,y) { this.x=x; this.y=y; this.range=100; this.cooldown=0; this.fireRate=500; }
  update(dt, enemies) { this.cooldown-=dt; if(this.cooldown>0) return;
    const target = enemies.find(e => Math.hypot(e.x-this.x, e.y-this.y) < this.range);
    if(target) { target.hp--; this.cooldown=this.fireRate; } } }""",
    },
    {
        "id": "game-particles",
        "scaffolds": ["game"],
        "keywords": ["particle", "explosion", "effect", "spark", "firework", "trail", "emit"],
        "always_for": [],
        "label": "Particle system (emit + fade + physics)",
        "code": """\
// Particle emitter: burst on event, fade over lifetime
class Particle {
  constructor(x, y, color) {
    this.x = x; this.y = y; this.color = color;
    const angle = Math.random() * Math.PI * 2, speed = Math.random() * 4 + 1;
    this.vx = Math.cos(angle) * speed; this.vy = Math.sin(angle) * speed;
    this.life = 1.0; this.decay = 0.02 + Math.random() * 0.02;
    this.size = Math.random() * 4 + 2;
  }
  update() { this.x += this.vx; this.y += this.vy; this.vy += 0.05; this.vx *= 0.98; this.life -= this.decay; }
  draw(ctx) { ctx.globalAlpha = this.life; ctx.fillStyle = this.color;
    ctx.beginPath(); ctx.arc(this.x, this.y, this.size * this.life, 0, Math.PI*2); ctx.fill(); ctx.globalAlpha = 1; }
  get dead() { return this.life <= 0; }
}
function emitBurst(x, y, color, count = 20) {
  for (let i = 0; i < count; i++) state.entities.push(new Particle(x, y, color));
}""",
    },
    {
        "id": "game-sprite",
        "scaffolds": ["game"],
        "keywords": ["sprite", "animation", "frame", "image", "spritesheet", "character", "walk"],
        "always_for": [],
        "label": "Sprite animation (frame cycling + spritesheet)",
        "code": """\
// Sprite with frame animation (works with or without spritesheet)
class Sprite extends Entity {
  constructor(x, y, w, h, frameCount) {
    super(x, y, w, h); this.frame = 0; this.frameCount = frameCount;
    this.frameTimer = 0; this.frameDuration = 100; // ms per frame
  }
  update(dt) { this.frameTimer += dt; if (this.frameTimer > this.frameDuration) { this.frame = (this.frame + 1) % this.frameCount; this.frameTimer = 0; } }
  draw(ctx) {
    // Without spritesheet: draw different shapes per frame
    ctx.fillStyle = ['#f44','#4f4','#44f','#ff4'][this.frame % 4];
    ctx.fillRect(this.x, this.y, this.w, this.h);
    // With spritesheet: ctx.drawImage(sheet, this.frame * this.w, 0, this.w, this.h, this.x, this.y, this.w, this.h);
  }
}""",
    },
    {
        "id": "game-camera",
        "scaffolds": ["game"],
        "keywords": ["camera", "scroll", "viewport", "world", "map", "level", "pan", "zoom", "follow"],
        "always_for": [],
        "label": "Camera/viewport (follow player + world bounds)",
        "code": """\
// Camera that follows player, clamps to world bounds
const camera = { x: 0, y: 0 };
const WORLD = { width: 3000, height: 2000 };
function updateCamera(target) {
  camera.x = target.x - canvas.width / 2;
  camera.y = target.y - canvas.height / 2;
  camera.x = Math.max(0, Math.min(WORLD.width - canvas.width, camera.x));
  camera.y = Math.max(0, Math.min(WORLD.height - canvas.height, camera.y));
}
function drawWithCamera(fn) { ctx.save(); ctx.translate(-camera.x, -camera.y); fn(); ctx.restore(); }
// Usage in loop: updateCamera(player); drawWithCamera(() => { entities.forEach(e => e.draw(ctx)); });""",
    },
    {
        "id": "game-audio",
        "scaffolds": ["game"],
        "keywords": ["sound", "audio", "music", "sfx", "beep", "tone", "volume"],
        "always_for": [],
        "label": "Audio/sound effects (Web Audio API)",
        "code": """\
// Simple sound system using Web Audio API (no files needed)
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
function playTone(freq, duration, type = 'square') {
  const osc = audioCtx.createOscillator(), gain = audioCtx.createGain();
  osc.type = type; osc.frequency.value = freq;
  gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + duration);
  osc.connect(gain); gain.connect(audioCtx.destination);
  osc.start(); osc.stop(audioCtx.currentTime + duration);
}
// Usage: playTone(440, 0.1) for shoot, playTone(200, 0.3, 'sawtooth') for explosion""",
    },
    {
        "id": "game-powerup",
        "scaffolds": ["game"],
        "keywords": ["power", "powerup", "collectible", "bonus", "coin", "pickup", "item", "health"],
        "always_for": [],
        "label": "Collectibles/power-ups (spawn + pickup + timer)",
        "code": """\
// Collectible pattern: spawn, float, pickup on collision, timed effect
class PowerUp extends Entity {
  constructor(x, y, type) { super(x, y, 24, 24); this.type = type; this.bobPhase = Math.random() * Math.PI * 2; }
  update(dt) { this.y += Math.sin(Date.now() / 300 + this.bobPhase) * 0.5; } // Bob animation
  draw(ctx) { ctx.fillStyle = this.type === 'speed' ? '#0f0' : this.type === 'shield' ? '#00f' : '#ff0';
    ctx.beginPath(); ctx.arc(this.x+12, this.y+12, 12, 0, Math.PI*2); ctx.fill(); }
}
// Apply effect on pickup
function applyPowerUp(player, powerup) {
  if (powerup.type === 'speed') { player.speed *= 1.5; setTimeout(() => player.speed /= 1.5, 5000); }
  if (powerup.type === 'health') { state.lives = Math.min(state.lives + 1, 5); }
  powerup.dead = true; state.score += 50;
}""",
    },
    {
        "id": "game-hud",
        "scaffolds": ["game"],
        "keywords": ["hud", "health", "bar", "score", "display", "status", "overlay", "ui", "hearts"],
        "always_for": [],
        "label": "HUD overlay (health bar + score + minimap)",
        "code": """\
// HUD drawn on canvas (not DOM) for game feel
function drawHUD(ctx) {
  ctx.save(); // HUD uses screen coords, not world coords
  // Health bar
  const barW = 200, barH = 16, hp = state.lives / 5;
  ctx.fillStyle = '#333'; ctx.fillRect(10, 10, barW, barH);
  ctx.fillStyle = hp > 0.3 ? '#0f0' : '#f00'; ctx.fillRect(10, 10, barW * hp, barH);
  ctx.strokeStyle = '#fff'; ctx.strokeRect(10, 10, barW, barH);
  // Score
  ctx.fillStyle = '#fff'; ctx.font = 'bold 20px monospace'; ctx.textAlign = 'right';
  ctx.fillText('SCORE: ' + state.score.toString().padStart(6, '0'), canvas.width - 10, 26);
  // Level
  ctx.textAlign = 'left'; ctx.fillText('LEVEL ' + state.level, 10, 46);
  ctx.restore();
}""",
    },
    {
        "id": "game-tilemap",
        "scaffolds": ["game"],
        "keywords": ["tile", "tilemap", "map", "level", "terrain", "dungeon", "maze", "wall"],
        "always_for": [],
        "label": "Tilemap rendering (2D array + collision)",
        "code": """\
// Tilemap: 2D array, render visible tiles, collision check
const TILE_SIZE = 32;
const tilemap = [ // 0=empty, 1=wall, 2=floor
  [1,1,1,1,1,1,1,1],
  [1,2,2,2,2,2,2,1],
  [1,2,1,2,2,1,2,1],
  [1,2,2,2,2,2,2,1],
  [1,1,1,1,1,1,1,1],
];
const TILE_COLORS = { 0: '#000', 1: '#666', 2: '#ddd' };
function drawTilemap(ctx) {
  for (let r = 0; r < tilemap.length; r++) for (let c = 0; c < tilemap[r].length; c++) {
    ctx.fillStyle = TILE_COLORS[tilemap[r][c]] || '#000';
    ctx.fillRect(c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1);
  }
}
function isSolid(x, y) { const r = Math.floor(y/TILE_SIZE), c = Math.floor(x/TILE_SIZE);
  return r < 0 || c < 0 || r >= tilemap.length || c >= tilemap[0].length || tilemap[r][c] === 1; }""",
    },
    {
        "id": "game-leaderboard",
        "scaffolds": ["game"],
        "keywords": ["leaderboard", "high", "score", "best", "rank", "record", "personal"],
        "always_for": [],
        "label": "Leaderboard/high scores (localStorage + sorted display)",
        "code": """\
// High score persistence and display
const SCORE_KEY = 'highscores_v1';
function getScores() { try { return JSON.parse(localStorage.getItem(SCORE_KEY)) || []; } catch { return []; } }
function saveScore(name, score) {
  const scores = getScores(); scores.push({ name, score, date: Date.now() });
  scores.sort((a, b) => b.score - a.score); // Highest first
  localStorage.setItem(SCORE_KEY, JSON.stringify(scores.slice(0, 10))); // Keep top 10
}
function renderLeaderboard(containerId) {
  const el = document.getElementById(containerId), scores = getScores();
  el.innerHTML = scores.length ? scores.map((s, i) =>
    '<div class="score-row"><span>#' + (i+1) + '</span><span>' + s.name + '</span><span>' + s.score + '</span></div>'
  ).join('') : '<p>No scores yet</p>';
}""",
    },
    {
        "id": "game-touch",
        "scaffolds": ["game"],
        "keywords": ["touch", "mobile", "swipe", "tap", "gesture", "joystick", "phone"],
        "always_for": [],
        "label": "Touch/mobile controls (virtual joystick + swipe)",
        "code": """\
// Touch controls: virtual joystick + tap-to-shoot
let touchDir = { x: 0, y: 0 }, touchActive = false;
canvas.addEventListener('touchstart', (e) => { e.preventDefault(); touchActive = true; handleTouch(e); });
canvas.addEventListener('touchmove', (e) => { e.preventDefault(); handleTouch(e); });
canvas.addEventListener('touchend', () => { touchActive = false; touchDir = { x: 0, y: 0 }; });
function handleTouch(e) {
  const t = e.touches[0], cx = canvas.width / 2, cy = canvas.height / 2;
  touchDir.x = (t.clientX - cx) / cx; // -1 to 1
  touchDir.y = (t.clientY - cy) / cy;
}
// Usage in player update: if (touchActive) { this.x += touchDir.x * speed; this.y += touchDir.y * speed; }""",
    },

    # ===== DASHBOARD — Additional =====
    {
        "id": "dashboard-table",
        "scaffolds": ["dashboard", "static"],
        "keywords": ["table", "sort", "column", "row", "data", "grid", "spreadsheet", "tabular"],
        "always_for": [],
        "label": "Sortable data table",
        "code": """\
// Sortable table: click header to sort, alternating row colors
function renderTable(containerId, data, columns) {
  let sortCol = null, sortAsc = true;
  function render() {
    const sorted = sortCol ? [...data].sort((a,b) => { const v = a[sortCol] > b[sortCol] ? 1 : -1; return sortAsc ? v : -v; }) : data;
    const esc = s => String(s).replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'})[c]);
    document.getElementById(containerId).innerHTML = '<table><thead><tr>' +
      columns.map(c => '<th data-col="' + c.key + '">' + esc(c.label) + (sortCol===c.key ? (sortAsc?' ▲':' ▼') : '') + '</th>').join('') +
      '</tr></thead><tbody>' + sorted.map(row => '<tr>' + columns.map(c => '<td>' + esc(row[c.key]) + '</td>').join('') + '</tr>').join('') + '</tbody></table>';
    document.querySelectorAll('#' + containerId + ' th').forEach(th => th.addEventListener('click', () => {
      if (sortCol === th.dataset.col) sortAsc = !sortAsc; else { sortCol = th.dataset.col; sortAsc = true; } render();
    }));
  }
  render();
}""",
    },
    {
        "id": "dashboard-gauge",
        "scaffolds": ["dashboard"],
        "keywords": ["gauge", "meter", "speedometer", "dial", "progress", "circular", "radial", "arc"],
        "always_for": [],
        "label": "Gauge/meter (CSS conic-gradient or canvas arc)",
        "code": """\
// CSS gauge using conic-gradient (no canvas needed)
function setGauge(elementId, value, max, label) {
  const pct = Math.min(100, (value / max) * 100);
  const color = pct > 75 ? '#ef4444' : pct > 50 ? '#f59e0b' : '#22c55e';
  const el = document.getElementById(elementId);
  el.style.background = 'conic-gradient(' + color + ' ' + pct + '%, #333 ' + pct + '%)';
  el.querySelector('.gauge-value').textContent = value;
  el.querySelector('.gauge-label').textContent = label;
}
/* CSS: .gauge { width: 120px; height: 120px; border-radius: 50%; display: flex; align-items: center;
  justify-content: center; position: relative; }
  .gauge::after { content: ''; position: absolute; inset: 12px; border-radius: 50%; background: var(--card); } */""",
    },
    {
        "id": "dashboard-export",
        "scaffolds": ["dashboard", "form", "static"],
        "keywords": ["export", "csv", "download", "save", "pdf", "print", "report"],
        "always_for": [],
        "label": "Export to CSV (data → download)",
        "code": """\
// Export array of objects to CSV download
function exportCSV(data, filename = 'export.csv') {
  if (!data.length) return;
  const keys = Object.keys(data[0]);
  const csv = [keys.join(','), ...data.map(row => keys.map(k => {
    const val = String(row[k] ?? '').replace(/"/g, '""');
    return val.includes(',') || val.includes('"') || val.includes('\\n') ? '"' + val + '"' : val;
  }).join(','))].join('\\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = filename; a.click(); URL.revokeObjectURL(a.href);
}""",
    },

    # ===== FORM — Additional =====
    {
        "id": "form-dragdrop",
        "scaffolds": ["form", "static"],
        "keywords": ["drag", "drop", "sortable", "reorder", "kanban", "column", "move", "draggable"],
        "always_for": [],
        "label": "Drag and drop (native HTML5 DnD API)",
        "code": """\
// HTML5 drag and drop: draggable items between containers
function initDragDrop(containerSelector) {
  document.querySelectorAll(containerSelector).forEach(container => {
    container.addEventListener('dragover', (e) => { e.preventDefault(); container.classList.add('drag-over'); });
    container.addEventListener('dragleave', () => container.classList.remove('drag-over'));
    container.addEventListener('drop', (e) => { e.preventDefault(); container.classList.remove('drag-over');
      const id = e.dataTransfer.getData('text/plain');
      const el = document.getElementById(id); if (el) container.appendChild(el);
    });
  });
  document.querySelectorAll('[draggable="true"]').forEach(item => {
    item.addEventListener('dragstart', (e) => { e.dataTransfer.setData('text/plain', item.id); item.classList.add('dragging'); });
    item.addEventListener('dragend', () => item.classList.remove('dragging'));
  });
}
/* CSS: .drag-over { outline: 2px dashed var(--accent); } .dragging { opacity: 0.5; } */""",
    },
    {
        "id": "form-file-upload",
        "scaffolds": ["form"],
        "keywords": ["upload", "file", "image", "preview", "drop zone", "attachment", "photo"],
        "always_for": [],
        "label": "File upload with preview",
        "code": """\
// File upload: drag zone + preview (images only for security)
function initFileUpload(dropZoneId, previewId) {
  const zone = document.getElementById(dropZoneId), preview = document.getElementById(previewId);
  const ALLOWED = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
  const MAX_SIZE = 5 * 1024 * 1024; // 5MB
  function handleFiles(files) {
    preview.innerHTML = '';
    Array.from(files).forEach(file => {
      if (!ALLOWED.includes(file.type)) { preview.innerHTML += '<p class="error">Invalid type: ' + file.name + '</p>'; return; }
      if (file.size > MAX_SIZE) { preview.innerHTML += '<p class="error">Too large: ' + file.name + '</p>'; return; }
      const reader = new FileReader();
      reader.onload = (e) => { const img = document.createElement('img'); img.src = e.target.result; img.style.maxWidth = '200px'; preview.appendChild(img); };
      reader.readAsDataURL(file);
    });
  }
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('active'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('active'));
  zone.addEventListener('drop', (e) => { e.preventDefault(); zone.classList.remove('active'); handleFiles(e.dataTransfer.files); });
}""",
    },
    {
        "id": "form-autocomplete",
        "scaffolds": ["form", "static"],
        "keywords": ["autocomplete", "typeahead", "suggest", "search", "combobox", "dropdown"],
        "always_for": [],
        "label": "Autocomplete/typeahead search",
        "code": """\
// Autocomplete: debounced input → filter → dropdown
function initAutocomplete(inputId, listId, items) {
  const input = document.getElementById(inputId), list = document.getElementById(listId);
  let timer = null;
  function render(matches) {
    const esc = s => s.replace(/[<>&"]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'})[c]);
    list.innerHTML = matches.map(m => '<div class="ac-item" data-value="' + esc(m) + '">' + esc(m) + '</div>').join('');
    list.classList.toggle('hidden', !matches.length);
  }
  input.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(() => {
    const q = input.value.toLowerCase().trim();
    render(q ? items.filter(i => i.toLowerCase().includes(q)).slice(0, 8) : []);
  }, 200); });
  list.addEventListener('click', (e) => { const item = e.target.closest('.ac-item');
    if (item) { input.value = item.dataset.value; list.classList.add('hidden'); } });
  document.addEventListener('click', (e) => { if (!e.target.closest('#'+inputId+',#'+listId)) list.classList.add('hidden'); });
}""",
    },
    {
        "id": "form-password",
        "scaffolds": ["form"],
        "keywords": ["password", "strength", "meter", "login", "register", "signup", "auth", "secure"],
        "always_for": [],
        "label": "Password strength meter",
        "code": """\
// Password strength: real-time feedback as user types
function initPasswordMeter(inputId, meterId) {
  const input = document.getElementById(inputId), meter = document.getElementById(meterId);
  function score(pw) {
    let s = 0; if (pw.length >= 8) s++; if (pw.length >= 12) s++;
    if (/[a-z]/.test(pw) && /[A-Z]/.test(pw)) s++; if (/\\d/.test(pw)) s++; if (/[^a-zA-Z0-9]/.test(pw)) s++;
    return Math.min(4, s);
  }
  const labels = ['Weak', 'Fair', 'Good', 'Strong', 'Excellent'];
  const colors = ['#ef4444', '#f59e0b', '#eab308', '#22c55e', '#16a34a'];
  input.addEventListener('input', () => {
    const s = score(input.value);
    meter.style.width = ((s + 1) * 20) + '%'; meter.style.background = colors[s];
    meter.textContent = labels[s]; meter.parentElement.title = labels[s];
  });
}""",
    },

    # ===== CROSS-CUTTING UI PATTERNS =====
    {
        "id": "ui-modal",
        "scaffolds": ["static", "form", "dashboard"],
        "keywords": ["modal", "dialog", "popup", "overlay", "confirm", "lightbox"],
        "always_for": [],
        "label": "Modal/dialog (accessible, Escape to close)",
        "code": """\
// Accessible modal: focus trap, Escape closes, backdrop click closes
function openModal(modalId) {
  const modal = document.getElementById(modalId); modal.classList.add('active'); modal.setAttribute('aria-hidden', 'false');
  const focusable = modal.querySelectorAll('button, input, select, textarea, a[href]');
  if (focusable.length) focusable[0].focus();
  modal._onKey = (e) => { if (e.key === 'Escape') closeModal(modalId); };
  modal._onBackdrop = (e) => { if (e.target === modal) closeModal(modalId); };
  document.addEventListener('keydown', modal._onKey); modal.addEventListener('click', modal._onBackdrop);
}
function closeModal(modalId) {
  const modal = document.getElementById(modalId); modal.classList.remove('active'); modal.setAttribute('aria-hidden', 'true');
  document.removeEventListener('keydown', modal._onKey); modal.removeEventListener('click', modal._onBackdrop);
}
/* CSS: .modal { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.5); align-items:center; justify-content:center; z-index:100; }
.modal.active { display:flex; } .modal-content { background:var(--card); border-radius:var(--radius); padding:2rem; max-width:500px; width:90%; } */""",
    },
    {
        "id": "ui-toast",
        "scaffolds": ["static", "form", "dashboard"],
        "keywords": ["toast", "notification", "alert", "snackbar", "message", "feedback", "flash"],
        "always_for": [],
        "label": "Toast notifications (auto-dismiss + stack)",
        "code": """\
// Toast notification system: show, auto-dismiss, stack multiple
const toastContainer = document.createElement('div');
toastContainer.id = 'toast-container';
toastContainer.style.cssText = 'position:fixed;top:1rem;right:1rem;z-index:999;display:flex;flex-direction:column;gap:0.5rem;';
document.body.appendChild(toastContainer);
function showToast(msg, type = 'info', duration = 3000) {
  const esc = s => s.replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'})[c]);
  const el = document.createElement('div');
  const colors = { info: '#3b82f6', success: '#22c55e', error: '#ef4444', warning: '#f59e0b' };
  el.style.cssText = 'padding:0.75rem 1rem;border-radius:8px;color:white;background:' + (colors[type]||colors.info) + ';font-size:0.9rem;opacity:0;transform:translateX(100%);transition:all 0.3s;';
  el.textContent = msg;
  toastContainer.appendChild(el);
  requestAnimationFrame(() => { el.style.opacity = '1'; el.style.transform = 'none'; });
  setTimeout(() => { el.style.opacity = '0'; el.style.transform = 'translateX(100%)'; setTimeout(() => el.remove(), 300); }, duration);
}""",
    },
    {
        "id": "ui-tabs",
        "scaffolds": ["static", "form", "dashboard"],
        "keywords": ["tab", "tabs", "panel", "switch", "section", "view", "page", "navigation"],
        "always_for": [],
        "label": "Tab navigation (accessible, keyboard support)",
        "code": """\
// Accessible tabs: arrow key navigation, ARIA roles
function initTabs(containerSelector) {
  const container = document.querySelector(containerSelector);
  const buttons = container.querySelectorAll('[role="tab"]');
  const panels = container.querySelectorAll('[role="tabpanel"]');
  function activate(index) {
    buttons.forEach((b, i) => { b.setAttribute('aria-selected', i === index); b.tabIndex = i === index ? 0 : -1; });
    panels.forEach((p, i) => p.hidden = i !== index);
    buttons[index].focus();
  }
  buttons.forEach((btn, i) => {
    btn.addEventListener('click', () => activate(i));
    btn.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight') activate((i + 1) % buttons.length);
      if (e.key === 'ArrowLeft') activate((i - 1 + buttons.length) % buttons.length);
    });
  });
  activate(0);
}""",
    },
    {
        "id": "ui-accordion",
        "scaffolds": ["static", "form"],
        "keywords": ["accordion", "collapse", "expand", "faq", "dropdown", "toggle", "details"],
        "always_for": [],
        "label": "Accordion/collapsible sections",
        "code": """\
// Accordion: click to expand/collapse, only one open at a time
function initAccordion(selector, allowMultiple = false) {
  document.querySelectorAll(selector + ' .accordion-header').forEach(header => {
    header.addEventListener('click', () => {
      const item = header.parentElement, isOpen = item.classList.contains('open');
      if (!allowMultiple) document.querySelectorAll(selector + ' .accordion-item.open').forEach(i => i.classList.remove('open'));
      item.classList.toggle('open', !isOpen);
    });
  });
}
/* CSS: .accordion-body { max-height: 0; overflow: hidden; transition: max-height 0.3s; }
.accordion-item.open .accordion-body { max-height: 500px; }
.accordion-header { cursor: pointer; padding: 1rem; border-bottom: 1px solid var(--border); }
.accordion-header::after { content: '▸'; float: right; transition: transform 0.3s; }
.accordion-item.open .accordion-header::after { transform: rotate(90deg); } */""",
    },
    {
        "id": "ui-theme-toggle",
        "scaffolds": ["static", "form", "dashboard"],
        "keywords": ["dark", "light", "theme", "mode", "toggle", "switch", "appearance"],
        "always_for": [],
        "label": "Dark/light theme toggle (CSS vars + localStorage)",
        "code": """\
// Theme toggle: swap CSS custom properties, persist choice
function initTheme() {
  const saved = localStorage.getItem('theme') || (matchMedia('(prefers-color-scheme:dark)').matches ? 'dark' : 'light');
  document.documentElement.dataset.theme = saved;
}
function toggleTheme() {
  const current = document.documentElement.dataset.theme;
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  localStorage.setItem('theme', next);
}
/* CSS: :root, [data-theme="light"] { --bg: #fff; --text: #1a1a1a; --card: #f5f5f5; }
[data-theme="dark"] { --bg: #0f172a; --text: #f1f5f9; --card: #1e293b; }
body { background: var(--bg); color: var(--text); transition: background 0.3s, color 0.3s; } */""",
    },
    {
        "id": "ui-responsive-nav",
        "scaffolds": ["static", "form", "dashboard"],
        "keywords": ["navbar", "hamburger", "menu", "responsive", "mobile", "navigation", "sidebar"],
        "always_for": [],
        "label": "Responsive navbar (hamburger menu on mobile)",
        "code": """\
// Responsive nav: full on desktop, hamburger on mobile
document.getElementById('menu-toggle').addEventListener('click', () => {
  document.getElementById('nav-links').classList.toggle('open');
});
// Close on link click (mobile)
document.querySelectorAll('#nav-links a').forEach(a => a.addEventListener('click', () => {
  document.getElementById('nav-links').classList.remove('open');
}));
/* CSS: .nav { display: flex; justify-content: space-between; align-items: center; padding: 1rem; }
.nav-links { display: flex; gap: 1rem; list-style: none; }
.menu-toggle { display: none; background: none; border: none; font-size: 1.5rem; cursor: pointer; color: var(--text); }
@media (max-width: 768px) {
  .menu-toggle { display: block; }
  .nav-links { display: none; flex-direction: column; position: absolute; top: 60px; left: 0; right: 0; background: var(--bg); padding: 1rem; }
  .nav-links.open { display: flex; } } */""",
    },
    {
        "id": "ui-infinite-scroll",
        "scaffolds": ["static", "form"],
        "keywords": ["infinite", "scroll", "load more", "pagination", "lazy", "feed", "endless"],
        "always_for": [],
        "label": "Infinite scroll (intersection observer)",
        "code": """\
// Infinite scroll using IntersectionObserver (no scroll event listeners)
let page = 1, loading = false;
const sentinel = document.getElementById('scroll-sentinel');
const observer = new IntersectionObserver((entries) => {
  if (entries[0].isIntersecting && !loading) { loading = true; loadMore(); }
}, { rootMargin: '200px' });
observer.observe(sentinel);
function loadMore() {
  // Simulate async data fetch
  setTimeout(() => {
    const items = Array.from({ length: 10 }, (_, i) => ({ id: page * 10 + i, text: 'Item ' + (page * 10 + i) }));
    const container = document.getElementById('feed');
    items.forEach(item => { const el = document.createElement('div'); el.className = 'feed-item';
      el.textContent = item.text; container.insertBefore(el, sentinel); });
    page++; loading = false;
  }, 500);
}""",
    },

    # ===== DATA & STATE PATTERNS =====
    {
        "id": "data-fetch",
        "scaffolds": ["static", "dashboard", "form"],
        "keywords": ["fetch", "api", "ajax", "request", "http", "rest", "json", "endpoint", "load"],
        "always_for": [],
        "label": "Fetch API with error handling + loading state",
        "code": """\
// Fetch pattern: loading state, error handling, retry
async function fetchData(url, opts = {}) {
  const el = document.getElementById('status');
  el.textContent = 'Loading...'; el.className = 'status loading';
  try {
    const resp = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...opts });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    el.textContent = ''; el.className = 'status';
    return data;
  } catch (err) {
    el.textContent = 'Error: ' + err.message; el.className = 'status error';
    return null;
  }
}
// Usage: const users = await fetchData('/api/users');""",
    },
    {
        "id": "data-debounce",
        "scaffolds": ["static", "form", "dashboard"],
        "keywords": ["debounce", "throttle", "delay", "rate", "limit", "search"],
        "always_for": [],
        "label": "Debounce + throttle utilities",
        "code": """\
// Debounce: delay execution until pause in calls
function debounce(fn, ms) { let timer; return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); }; }
// Throttle: execute at most once per interval
function throttle(fn, ms) { let last = 0; return (...args) => { const now = Date.now(); if (now - last >= ms) { last = now; fn(...args); } }; }
// Usage: input.addEventListener('input', debounce(handleSearch, 300));
// Usage: window.addEventListener('scroll', throttle(handleScroll, 100));""",
    },
    {
        "id": "data-state-manager",
        "scaffolds": ["static", "form"],
        "keywords": ["state", "store", "redux", "reactive", "subscribe", "observe", "bind"],
        "always_for": [],
        "label": "Simple state manager (subscribe + render on change)",
        "code": """\
// Minimal reactive state: set() triggers all subscribers
function createStore(initial) {
  let state = { ...initial };
  const listeners = new Set();
  return {
    get: (key) => state[key],
    getAll: () => ({ ...state }),
    set: (key, value) => { state[key] = value; listeners.forEach(fn => fn(state)); },
    update: (partial) => { Object.assign(state, partial); listeners.forEach(fn => fn(state)); },
    subscribe: (fn) => { listeners.add(fn); return () => listeners.delete(fn); },
  };
}
// Usage: const store = createStore({ count: 0, items: [] });
// store.subscribe(state => { document.getElementById('count').textContent = state.count; });
// store.set('count', store.get('count') + 1);""",
    },
    {
        "id": "data-undo-redo",
        "scaffolds": ["form", "static"],
        "keywords": ["undo", "redo", "history", "revert", "command", "ctrl-z"],
        "always_for": [],
        "label": "Undo/redo history stack",
        "code": """\
// Undo/redo: snapshot state on each action
function createHistory(initial) {
  const stack = [JSON.parse(JSON.stringify(initial))];
  let idx = 0;
  return {
    push(state) { stack.splice(idx + 1); stack.push(JSON.parse(JSON.stringify(state))); idx = stack.length - 1; },
    undo() { if (idx > 0) idx--; return JSON.parse(JSON.stringify(stack[idx])); },
    redo() { if (idx < stack.length - 1) idx++; return JSON.parse(JSON.stringify(stack[idx])); },
    canUndo() { return idx > 0; },
    canRedo() { return idx < stack.length - 1; },
  };
}
// Keyboard shortcut: document.addEventListener('keydown', (e) => {
//   if (e.ctrlKey && e.key === 'z') { state = history.undo(); render(); }
//   if (e.ctrlKey && e.key === 'y') { state = history.redo(); render(); } });""",
    },

    # ===== ANIMATION & VISUAL =====
    {
        "id": "anim-transition",
        "scaffolds": ["static", "form", "dashboard"],
        "keywords": ["animate", "animation", "transition", "fade", "slide", "smooth", "motion", "ease"],
        "always_for": [],
        "label": "CSS transitions + JS animation helpers",
        "code": """\
// Animate element: add class, wait for transition, remove
function animate(el, className, duration = 300) {
  return new Promise(resolve => { el.classList.add(className); setTimeout(() => { el.classList.remove(className); resolve(); }, duration); });
}
// Fade in/out
function fadeIn(el, ms = 300) { el.style.opacity = '0'; el.style.display = ''; requestAnimationFrame(() => { el.style.transition = 'opacity ' + ms + 'ms'; el.style.opacity = '1'; }); }
function fadeOut(el, ms = 300) { el.style.transition = 'opacity ' + ms + 'ms'; el.style.opacity = '0'; setTimeout(() => el.style.display = 'none', ms); }
// Counter animation (number counting up)
function animateCount(el, target, duration = 1000) {
  const start = parseInt(el.textContent) || 0, range = target - start, startTime = performance.now();
  function step(now) { const pct = Math.min(1, (now - startTime) / duration);
    el.textContent = Math.round(start + range * pct).toLocaleString();
    if (pct < 1) requestAnimationFrame(step); }
  requestAnimationFrame(step);
}""",
    },
    {
        "id": "anim-scroll-reveal",
        "scaffolds": ["static"],
        "keywords": ["scroll", "reveal", "appear", "intersection", "lazy", "viewport", "parallax"],
        "always_for": [],
        "label": "Scroll-triggered reveal animations",
        "code": """\
// Reveal elements as they scroll into view
function initScrollReveal(selector = '.reveal') {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('visible'); observer.unobserve(e.target); } });
  }, { threshold: 0.15 });
  document.querySelectorAll(selector).forEach(el => observer.observe(el));
}
/* CSS: .reveal { opacity: 0; transform: translateY(30px); transition: opacity 0.6s, transform 0.6s; }
.reveal.visible { opacity: 1; transform: none; }
.reveal:nth-child(2) { transition-delay: 0.1s; } .reveal:nth-child(3) { transition-delay: 0.2s; } */""",
    },

    # ===== ACCESSIBILITY =====
    {
        "id": "a11y-focus",
        "scaffolds": ["static", "form", "dashboard"],
        "keywords": ["accessibility", "a11y", "focus", "keyboard", "aria", "screen reader", "wcag", "accessible"],
        "always_for": [],
        "label": "Accessibility (focus management + ARIA + skip link)",
        "code": """\
// Accessibility essentials: skip link, focus trap, ARIA live region
// Skip link (first element in body)
// <a href="#main-content" class="skip-link">Skip to content</a>
// CSS: .skip-link { position:absolute; top:-40px; left:0; z-index:100; } .skip-link:focus { top:0; }

// Focus trap for modals/overlays
function trapFocus(container) {
  const focusable = container.querySelectorAll('button,input,select,textarea,a[href],[tabindex]:not([tabindex="-1"])');
  const first = focusable[0], last = focusable[focusable.length - 1];
  container.addEventListener('keydown', (e) => {
    if (e.key !== 'Tab') return;
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  });
}
// ARIA live region for dynamic updates
// <div id="announcer" aria-live="polite" aria-atomic="true" class="sr-only"></div>
function announce(msg) { document.getElementById('announcer').textContent = msg; }
// CSS: .sr-only { position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0,0,0,0); }""",
    },

    # ===== MEDIA =====
    {
        "id": "media-gallery",
        "scaffolds": ["static"],
        "keywords": ["gallery", "image", "photo", "lightbox", "carousel", "slideshow", "grid"],
        "always_for": [],
        "label": "Image gallery with lightbox",
        "code": """\
// Image gallery: grid + click to open lightbox
function initGallery(gridId, images) {
  const esc = s => s.replace(/[<>&"]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'})[c]);
  const grid = document.getElementById(gridId);
  grid.innerHTML = images.map((src, i) => '<img class="gallery-thumb" src="' + esc(src) + '" data-idx="' + i + '" loading="lazy">').join('');
  // Lightbox
  const lb = document.createElement('div'); lb.className = 'lightbox hidden';
  lb.innerHTML = '<img class="lightbox-img"><button class="lightbox-close">&times;</button>';
  document.body.appendChild(lb);
  grid.addEventListener('click', (e) => { const img = e.target.closest('.gallery-thumb');
    if (img) { lb.querySelector('img').src = images[img.dataset.idx]; lb.classList.remove('hidden'); } });
  lb.addEventListener('click', () => lb.classList.add('hidden'));
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') lb.classList.add('hidden'); });
}
/* CSS: .gallery-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 0.5rem; }
.lightbox { position:fixed; inset:0; background:rgba(0,0,0,0.9); display:flex; align-items:center; justify-content:center; z-index:100; } */""",
    },
    {
        "id": "media-audio-player",
        "scaffolds": ["static"],
        "keywords": ["audio", "player", "music", "playlist", "play", "pause", "volume", "track"],
        "always_for": [],
        "label": "Audio player controls",
        "code": """\
// Custom audio player with progress bar and volume
function initPlayer(audioId, progressId, playBtnId, volumeId) {
  const audio = document.getElementById(audioId), progress = document.getElementById(progressId);
  const playBtn = document.getElementById(playBtnId), volume = document.getElementById(volumeId);
  playBtn.addEventListener('click', () => { if (audio.paused) { audio.play(); playBtn.textContent = '⏸'; } else { audio.pause(); playBtn.textContent = '▶'; } });
  audio.addEventListener('timeupdate', () => { progress.value = (audio.currentTime / audio.duration) * 100 || 0; });
  progress.addEventListener('input', () => { audio.currentTime = (progress.value / 100) * audio.duration; });
  volume.addEventListener('input', () => { audio.volume = volume.value / 100; });
  audio.addEventListener('ended', () => { playBtn.textContent = '▶'; progress.value = 0; });
}""",
    },
    {
        "id": "media-countdown",
        "scaffolds": ["static", "form"],
        "keywords": ["countdown", "timer", "clock", "stopwatch", "time", "pomodoro", "elapsed"],
        "always_for": [],
        "label": "Countdown timer / stopwatch",
        "code": """\
// Countdown timer with start/pause/reset
function initTimer(displayId, startBtnId, resetBtnId, totalSeconds) {
  let remaining = totalSeconds, interval = null, running = false;
  const display = document.getElementById(displayId);
  function format(s) { const m = Math.floor(s/60); return String(m).padStart(2,'0') + ':' + String(s%60).padStart(2,'0'); }
  function tick() { if (remaining <= 0) { clearInterval(interval); running = false; display.textContent = '00:00'; return; }
    remaining--; display.textContent = format(remaining); }
  function render() { display.textContent = format(remaining); }
  document.getElementById(startBtnId).addEventListener('click', () => {
    if (running) { clearInterval(interval); running = false; } else { interval = setInterval(tick, 1000); running = true; }
  });
  document.getElementById(resetBtnId).addEventListener('click', () => { clearInterval(interval); running = false; remaining = totalSeconds; render(); });
  render();
}""",
    },
    {
        "id": "media-markdown",
        "scaffolds": ["static", "form"],
        "keywords": ["markdown", "render", "format", "rich text", "parse", "wysiwyg", "editor"],
        "always_for": [],
        "label": "Simple markdown renderer (no dependencies)",
        "code": """\
// Minimal markdown to HTML (covers 90% of use cases, no XSS)
function renderMarkdown(md) {
  let html = md
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;') // Escape HTML first
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
    .replace(/\\*(.+?)\\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\\/li>)/s, '<ul>$1</ul>')
    .replace(/\\n\\n/g, '</p><p>')
    .replace(/\\n/g, '<br>');
  return '<p>' + html + '</p>';
}""",
    },

    # ===== GAME — Remaining =====
    {"id": "game-runner", "scaffolds": ["game"], "keywords": ["runner", "endless", "auto", "scroll", "obstacle", "dodge", "flappy", "dino"], "always_for": [], "label": "Endless runner (auto-scroll + obstacle dodge)",
     "code": "let scrollSpeed = 3, distance = 0;\nclass Obstacle extends Entity {\n  constructor() { super(canvas.width, canvas.height - 80, 30, 60); }\n  update(dt) { this.x -= scrollSpeed; if (this.x + this.w < 0) this.dead = true; }\n  draw(ctx) { ctx.fillStyle = '#f44'; super.draw(ctx); }\n}\nfunction updateRunner(dt) {\n  distance += scrollSpeed; scrollSpeed = 3 + distance * 0.0001;\n  if (Math.random() < 0.01 * scrollSpeed) state.entities.push(new Obstacle());\n  document.getElementById('distance').textContent = Math.floor(distance);\n}"},
    {"id": "game-turnbased", "scaffolds": ["game"], "keywords": ["turn", "card", "deck", "hand", "rpg", "battle", "strategy", "chess"], "always_for": [], "label": "Turn-based / card game (state machine + click actions)",
     "code": "const battle = { turn: 'player', playerHP: 100, enemyHP: 80, log: [] };\nfunction playerAttack() {\n  if (battle.turn !== 'player') return;\n  const dmg = Math.floor(Math.random() * 20) + 5;\n  battle.enemyHP -= dmg; battle.log.push('You deal ' + dmg + ' damage');\n  if (battle.enemyHP <= 0) { battle.log.push('Victory!'); renderBattle(); return; }\n  battle.turn = 'enemy'; renderBattle(); setTimeout(enemyTurn, 1000);\n}\nfunction enemyTurn() {\n  const dmg = Math.floor(Math.random() * 15) + 3;\n  battle.playerHP -= dmg; battle.log.push('Enemy deals ' + dmg);\n  battle.turn = 'player'; renderBattle();\n}"},
    {"id": "game-inventory", "scaffolds": ["game"], "keywords": ["inventory", "item", "equip", "slot", "backpack", "loot", "weapon"], "always_for": [], "label": "Inventory / equipment (grid slots + equip)",
     "code": "const inventory = { slots: new Array(20).fill(null), equipped: { weapon: null, armor: null } };\nfunction addItem(item) { const idx = inventory.slots.indexOf(null); if (idx === -1) return false; inventory.slots[idx] = { ...item }; return true; }\nfunction equipItem(i) { const item = inventory.slots[i]; if (!item?.slot) return; const prev = inventory.equipped[item.slot]; inventory.equipped[item.slot] = item; inventory.slots[i] = prev; }"},
    {"id": "game-dialogue", "scaffolds": ["game"], "keywords": ["dialogue", "text", "story", "npc", "conversation", "speech", "narrative"], "always_for": [], "label": "Dialogue / text box (typewriter + choices)",
     "code": "class DialogueBox {\n  constructor(el) { this.el = el; }\n  show(text, choices, onChoice) {\n    this.el.classList.remove('hidden'); let i = 0;\n    const p = this.el.querySelector('.text'); p.textContent = '';\n    const type = () => { if (i < text.length) { p.textContent += text[i++]; setTimeout(type, 30); }\n      else if (choices.length) { this.el.innerHTML += choices.map((c,j) => '<button data-idx=\"'+j+'\">'+c+'</button>').join('');\n        this.el.querySelectorAll('button').forEach(b => b.onclick = () => { onChoice(+b.dataset.idx); this.close(); }); }\n      else { this.el.onclick = () => this.close(); } };\n    type();\n  }\n  close() { this.el.classList.add('hidden'); }\n}"},
    {"id": "game-screentransition", "scaffolds": ["game"], "keywords": ["transition", "fade", "wipe", "scene", "level"], "always_for": [], "label": "Screen fade transitions between states",
     "code": "let fadeAlpha = 0, fading = false;\nfunction transitionTo(phase, cb) {\n  fading = true;\n  function out() { fadeAlpha += 0.05; if (fadeAlpha >= 1) { state.phase = phase; if (cb) cb(); inn(); } else requestAnimationFrame(out); }\n  function inn() { fadeAlpha -= 0.05; if (fadeAlpha <= 0) { fadeAlpha = 0; fading = false; } else requestAnimationFrame(inn); }\n  out();\n}\nfunction drawFade(ctx) { if (fadeAlpha > 0) { ctx.fillStyle = 'rgba(0,0,0,'+fadeAlpha+')'; ctx.fillRect(0,0,canvas.width,canvas.height); } }"},

    # ===== DASHBOARD — Remaining =====
    {"id": "dashboard-sparkline", "scaffolds": ["dashboard"], "keywords": ["sparkline", "mini", "inline", "trend", "compact"], "always_for": [], "label": "Sparkline / mini inline chart",
     "code": "function sparkline(canvasId, data, color='#3b82f6') {\n  const c = document.getElementById(canvasId), ctx = c.getContext('2d');\n  const w = c.width, h = c.height, max = Math.max(...data), min = Math.min(...data), range = max-min||1;\n  ctx.clearRect(0,0,w,h); ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;\n  data.forEach((v,i) => { const x = i/(data.length-1)*w, y = h-((v-min)/range)*(h-4)-2; i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y); });\n  ctx.stroke();\n}"},
    {"id": "dashboard-heatmap", "scaffolds": ["dashboard"], "keywords": ["heatmap", "calendar", "activity", "contribution", "github"], "always_for": [], "label": "Activity heatmap (CSS grid + color scale)",
     "code": "function renderHeatmap(id, data) {\n  const max = Math.max(...data.map(d=>d.value),1);\n  const colors = ['#1a1a2e','#16213e','#0f3460','#533483','#e94560'];\n  document.getElementById(id).innerHTML = data.map(d => {\n    const lvl = Math.min(4, Math.floor((d.value/max)*4));\n    return '<div class=\"hm-cell\" title=\"'+d.date+': '+d.value+'\" style=\"background:'+colors[lvl]+'\"></div>';\n  }).join('');\n}\n/* .heatmap{display:grid;grid-template-columns:repeat(52,12px);gap:2px} .hm-cell{width:12px;height:12px;border-radius:2px} */"},
    {"id": "dashboard-notification", "scaffolds": ["dashboard", "static"], "keywords": ["badge", "count", "notification", "bell", "unread"], "always_for": [], "label": "Notification badge / counter",
     "code": "function updateBadge(id, count) {\n  const el = document.getElementById(id);\n  el.textContent = count > 99 ? '99+' : count;\n  el.classList.toggle('hidden', count === 0);\n}\n/* .badge{position:absolute;top:-4px;right:-4px;background:#ef4444;color:#fff;font-size:0.7rem;min-width:18px;height:18px;border-radius:9px;display:flex;align-items:center;justify-content:center} */"},
    {"id": "dashboard-progress", "scaffolds": ["dashboard", "form", "static"], "keywords": ["progress", "step", "stepper", "pipeline", "workflow"], "always_for": [], "label": "Progress stepper / pipeline tracker",
     "code": "function renderStepper(id, steps, current) {\n  document.getElementById(id).innerHTML = steps.map((s,i) => {\n    const cls = i<current?'complete':i===current?'active':'pending';\n    return '<div class=\"step '+cls+'\"><span class=\"step-num\">'+(i+1)+'</span><span>'+s+'</span></div>';\n  }).join('<div class=\"step-line\"></div>');\n}\n/* .step-num{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:var(--border)} .step.active .step-num{background:var(--accent);color:#fff} */"},
    {"id": "dashboard-tree", "scaffolds": ["dashboard", "static"], "keywords": ["tree", "hierarchy", "folder", "nested", "directory"], "always_for": [], "label": "Collapsible tree / file browser",
     "code": "function renderTree(id, nodes, depth=0) {\n  return nodes.map(n => {\n    const has = n.children?.length;\n    return '<div style=\"padding-left:'+(depth*20)+'px\" class=\"tree-node\">'+\n      '<span class=\"toggle\">'+(has?'\\u25B6':'\\u00B7')+'</span> '+n.label+'</div>'+\n      (has?'<div class=\"tree-children hidden\">'+renderTree(id,n.children,depth+1)+'</div>':'');\n  }).join('');\n}\n// Click handler toggles .hidden on next sibling .tree-children"},

    # ===== FORM — Remaining =====
    {"id": "form-datepicker", "scaffolds": ["form"], "keywords": ["date", "picker", "calendar", "month", "schedule", "booking"], "always_for": [], "label": "Date picker (calendar grid, no library)",
     "code": "function initDatePicker(inputId, calId) {\n  let current = new Date(), selected = null;\n  function render() {\n    const y=current.getFullYear(), m=current.getMonth(), first=new Date(y,m,1).getDay(), days=new Date(y,m+1,0).getDate();\n    let html='<button data-dir=\"-1\">&lt;</button>'+['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m]+' '+y+'<button data-dir=\"1\">&gt;</button><br>';\n    for(let i=0;i<first;i++) html+='<span></span>'; for(let d=1;d<=days;d++) html+='<button data-day=\"'+d+'\">'+d+'</button>';\n    document.getElementById(calId).innerHTML=html;\n  }\n  document.getElementById(calId).addEventListener('click',e=>{if(e.target.dataset.dir){current.setMonth(current.getMonth()+ +e.target.dataset.dir);render()}\n    if(e.target.dataset.day){document.getElementById(inputId).value=current.getFullYear()+'-'+String(current.getMonth()+1).padStart(2,'0')+'-'+e.target.dataset.day.padStart(2,'0')}}); render();\n}"},
    {"id": "form-rating", "scaffolds": ["form", "static"], "keywords": ["rating", "star", "review", "vote", "score"], "always_for": [], "label": "Star rating (click + hover)",
     "code": "function initRating(id, max=5, onChange) {\n  const el=document.getElementById(id); let val=0;\n  el.innerHTML=Array.from({length:max},(_,i)=>'<button class=\"star\" data-v=\"'+(i+1)+'\">&#9733;</button>').join('');\n  function set(v,preview){el.querySelectorAll('.star').forEach((s,i)=>{s.classList.toggle('active',i<v);s.classList.toggle('preview',preview&&i<v)})}\n  el.onmouseover=e=>{const b=e.target.closest('.star');if(b)set(+b.dataset.v,true)};\n  el.onmouseout=()=>set(val,false);\n  el.onclick=e=>{const b=e.target.closest('.star');if(b){val=+b.dataset.v;set(val,false);if(onChange)onChange(val)}};\n}\n/* .star{background:none;border:none;font-size:1.5rem;color:#ccc;cursor:pointer} .star.active{color:#f59e0b} */"},
    {
        "id": "form-colorpicker",
        "scaffolds": ["form"],
        "keywords": ["color", "picker", "palette", "swatch", "hex"],
        "always_for": [],
        "label": "Color picker (swatches + native input)",
        "code": """\
// Color picker — preset swatches + a native <input type="color"> for the
// long tail. Clicking a swatch or the native input both drive the same
// hidden input, which is what downstream code reads/submits.
function initColorPicker(id, inputId) {
  const presets = ['#ef4444', '#f97316', '#eab308', '#22c55e',
                   '#3b82f6', '#8b5cf6', '#ec4899', '#000', '#fff'];
  const el = document.getElementById(id);
  el.innerHTML =
    presets.map((c) =>
      '<button class="swatch" data-c="' + c + '" style="background:' + c + '"></button>'
    ).join('') +
    '<input type="color">';

  el.addEventListener('click', (e) => {
    const swatch = e.target.closest('.swatch');
    if (swatch) document.getElementById(inputId).value = swatch.dataset.c;
  });
  el.querySelector('input[type=color]').addEventListener('input', (e) => {
    document.getElementById(inputId).value = e.target.value;
  });
}""",
    },
    {"id": "form-tags", "scaffolds": ["form"], "keywords": ["tag", "chip", "label", "multi", "keyword", "pill"], "always_for": [], "label": "Tag / chip input (add/remove + keyboard)",
     "code": "function initTags(inputId, containerId, onChange) {\n  const input=document.getElementById(inputId), box=document.getElementById(containerId); let tags=[];\n  function render(){box.innerHTML=tags.map((t,i)=>'<span class=\"tag\">'+t.replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'})[c])+' <button data-i=\"'+i+'\">&times;</button></span>').join('');if(onChange)onChange([...tags])}\n  input.onkeydown=e=>{if(e.key==='Enter'&&input.value.trim()){e.preventDefault();if(!tags.includes(input.value.trim())){tags.push(input.value.trim());render()}input.value=''}\n    if(e.key==='Backspace'&&!input.value&&tags.length){tags.pop();render()}};\n  box.onclick=e=>{const b=e.target.closest('[data-i]');if(b){tags.splice(+b.dataset.i,1);render()}};\n}"},
    {"id": "form-range", "scaffolds": ["form"], "keywords": ["range", "slider", "volume", "brightness", "level"], "always_for": [], "label": "Range slider with live value",
     "code": "function initRange(sliderId, valueId, unit='') {\n  const s=document.getElementById(sliderId), v=document.getElementById(valueId);\n  function update(){v.textContent=s.value+unit; const pct=((s.value-s.min)/(s.max-s.min))*100;\n    s.style.background='linear-gradient(to right,var(--accent) '+pct+'%,var(--border) '+pct+'%)';}\n  s.addEventListener('input',update); update();\n}"},
    {
        "id": "form-clipboard",
        "scaffolds": ["form", "static"],
        "keywords": ["clipboard", "copy", "paste", "share"],
        "always_for": [],
        "label": "Copy to clipboard button (modern API + execCommand fallback)",
        "code": """\
// Copy-to-clipboard with a "Copied!" affordance. Uses the modern
// navigator.clipboard API first; falls back to execCommand for older
// browsers or insecure contexts (file://, http://) where clipboard is
// blocked.
function initCopy(btnId, targetId) {
  const btn = document.getElementById(btnId);
  btn.addEventListener('click', async () => {
    const target = document.getElementById(targetId);
    const text = target.textContent || target.value;
    try {
      await navigator.clipboard.writeText(text);
      flashFeedback(btn, 'Copied!');
    } catch {
      // Fallback path — briefly create a hidden textarea to select+copy.
      const t = document.createElement('textarea');
      t.value = text;
      document.body.appendChild(t);
      t.select();
      document.execCommand('copy');
      t.remove();
      flashFeedback(btn, 'Copied!');
    }
  });
}

function flashFeedback(btn, text, ms = 2000) {
  const original = btn.textContent;
  btn.textContent = text;
  setTimeout(() => { btn.textContent = original; }, ms);
}""",
    },

    # ===== STATIC — Remaining =====
    {"id": "static-blog", "scaffolds": ["static"], "keywords": ["blog", "article", "post", "content", "reading", "cms"], "always_for": [], "label": "Blog / article layout (readable typography)",
     "code": "/* .article{max-width:680px;margin:0 auto;padding:2rem;line-height:1.7;font-size:1.1rem}\n.article h1{font-size:2.2rem;line-height:1.2;margin-bottom:0.5rem}\n.article img{max-width:100%;height:auto;border-radius:var(--radius);margin:1.5rem 0}\n.article blockquote{border-left:3px solid var(--accent);padding-left:1rem;color:var(--text-dim)}\n.article code{background:var(--card);padding:2px 6px;border-radius:4px;font-size:0.9em} */"},
    {"id": "static-pricing", "scaffolds": ["static"], "keywords": ["pricing", "plan", "tier", "subscription", "compare"], "always_for": [], "label": "Pricing table (tier cards + highlight)",
     "code": "function renderPricing(id, plans) {\n  const esc=s=>String(s).replace(/[<>&\"]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;','\"':'&quot;'})[c]);\n  document.getElementById(id).innerHTML=plans.map(p=>\n    '<div class=\"price-card'+(p.featured?' featured':'')+'\">'+(p.featured?'<span class=\"badge\">Popular</span>':'')+\n    '<h3>'+esc(p.name)+'</h3><div class=\"price\">$'+p.price+'<span>/mo</span></div>'+\n    '<ul>'+p.features.map(f=>'<li>'+(f.ok?'\\u2713 ':'\\u2717 ')+esc(f.text)+'</li>').join('')+'</ul>'+\n    '<button>'+esc(p.cta||'Start')+'</button></div>').join('');\n}"},
    {"id": "static-carousel", "scaffolds": ["static"], "keywords": ["carousel", "slider", "testimonial", "quote", "rotate"], "always_for": [], "label": "Card carousel / slider (auto-advance + dots)",
     "code": "function initCarousel(id) {\n  const el=document.getElementById(id), slides=el.querySelectorAll('.slide'); let cur=0,iv;\n  function show(i){cur=((i%slides.length)+slides.length)%slides.length;\n    slides.forEach((s,j)=>s.style.transform='translateX('+((j-cur)*100)+'%)');\n    el.querySelectorAll('.dot').forEach((d,j)=>d.classList.toggle('active',j===cur));}\n  const dots=document.createElement('div');dots.className='dots';\n  slides.forEach((_,i)=>{const d=document.createElement('button');d.className='dot';d.onclick=()=>show(i);dots.appendChild(d)});\n  el.appendChild(dots); show(0);\n  iv=setInterval(()=>show(cur+1),4000); el.onmouseenter=()=>clearInterval(iv); el.onmouseleave=()=>{iv=setInterval(()=>show(cur+1),4000)};\n}"},
    {"id": "static-faq", "scaffolds": ["static"], "keywords": ["faq", "question", "answer", "help", "support"], "always_for": [], "label": "FAQ (searchable accordion)",
     "code": "function initFAQ(id, searchId) {\n  const items=document.querySelectorAll('#'+id+' .faq-item');\n  items.forEach(item=>item.querySelector('.faq-q').onclick=()=>{const was=item.classList.contains('open');items.forEach(i=>i.classList.remove('open'));if(!was)item.classList.add('open')});\n  if(searchId) document.getElementById(searchId).oninput=e=>{const q=e.target.value.toLowerCase();items.forEach(i=>i.style.display=!q||i.textContent.toLowerCase().includes(q)?'':'none')};\n}"},
    {"id": "static-timeline", "scaffolds": ["static"], "keywords": ["timeline", "feed", "history", "events", "activity"], "always_for": [], "label": "Timeline / activity feed",
     "code": "function renderTimeline(id, events) {\n  const esc=s=>s.replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'})[c]);\n  document.getElementById(id).innerHTML=events.map(e=>\n    '<div class=\"tl-item\"><div class=\"tl-dot\"></div><div class=\"tl-content\"><span class=\"tl-date\">'+esc(e.date)+'</span><h3>'+esc(e.title)+'</h3><p>'+esc(e.text)+'</p></div></div>').join('');\n}\n/* .timeline{position:relative;padding-left:30px} .timeline::before{content:'';position:absolute;left:14px;top:0;bottom:0;width:2px;background:var(--border)} .tl-dot{position:absolute;left:-24px;width:12px;height:12px;border-radius:50%;background:var(--accent)} */"},
    {"id": "static-error", "scaffolds": ["static"], "keywords": ["404", "error", "not found", "empty", "offline"], "always_for": [], "label": "404 / error page",
     "code": "/* .error-page{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:80vh;text-align:center}\n.error-code{font-size:8rem;font-weight:900;color:var(--accent);opacity:0.3;line-height:1}\n.error-page h1{font-size:1.5rem;margin-top:0} .error-page p{color:var(--text-dim);max-width:400px}\n.error-btn{margin-top:1.5rem;padding:0.75rem 2rem;background:var(--accent);color:#fff;border-radius:var(--radius);text-decoration:none} */"},

    # ===== UI — Remaining =====
    {"id": "ui-tooltip", "scaffolds": ["static", "form", "dashboard"], "keywords": ["tooltip", "popover", "hint", "hover", "info"], "always_for": [], "label": "Tooltip (CSS-only + JS positioning)",
     "code": "/* CSS: [data-tooltip]{position:relative;cursor:help} [data-tooltip]::after{content:attr(data-tooltip);position:absolute;bottom:100%;left:50%;transform:translateX(-50%);padding:4px 8px;background:#1a1a1a;color:#fff;font-size:0.8rem;border-radius:4px;white-space:nowrap;opacity:0;pointer-events:none;transition:opacity 0.2s} [data-tooltip]:hover::after{opacity:1} */\n// Usage: <span data-tooltip=\"Help text\">Hover me</span>"},
    {"id": "ui-skeleton", "scaffolds": ["static", "dashboard", "form"], "keywords": ["skeleton", "loading", "placeholder", "shimmer"], "always_for": [], "label": "Skeleton loading (shimmer animation)",
     "code": "function showSkeleton(id, n=3){document.getElementById(id).innerHTML=Array.from({length:n},()=>'<div class=\"sk-card\"><div class=\"sk-line wide\"></div><div class=\"sk-line\"></div><div class=\"sk-line short\"></div></div>').join('')}\n/* .sk-line{height:14px;border-radius:4px;background:linear-gradient(90deg,var(--border) 25%,var(--card) 50%,var(--border) 75%);background-size:200% 100%;animation:shimmer 1.5s infinite;margin-bottom:8px}\n.sk-line.wide{width:80%;height:18px} .sk-line.short{width:40%} @keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}} */"},
    {"id": "ui-contextmenu", "scaffolds": ["static", "form"], "keywords": ["context", "right-click", "menu", "action"], "always_for": [], "label": "Context menu (right-click custom)",
     "code": "function initContextMenu(sel, items) {\n  const menu=document.createElement('div');menu.className='ctx-menu hidden';\n  menu.innerHTML=items.map((it,i)=>it.divider?'<hr>':'<button data-i=\"'+i+'\">'+it.label+'</button>').join('');\n  document.body.appendChild(menu);\n  document.querySelectorAll(sel).forEach(el=>el.oncontextmenu=e=>{e.preventDefault();menu.style.left=e.clientX+'px';menu.style.top=e.clientY+'px';menu.classList.remove('hidden');menu._t=el});\n  menu.onclick=e=>{const b=e.target.closest('[data-i]');if(b){items[+b.dataset.i]?.action?.(menu._t);menu.classList.add('hidden')}};\n  document.onclick=()=>menu.classList.add('hidden');\n}"},
    {"id": "ui-breadcrumbs", "scaffolds": ["static", "dashboard"], "keywords": ["breadcrumb", "path", "trail", "navigate"], "always_for": [], "label": "Breadcrumbs navigation",
     "code": "function renderBreadcrumbs(id, items) {\n  const esc=s=>s.replace(/[<>&\"]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;','\"':'&quot;'})[c]);\n  document.getElementById(id).innerHTML='<nav aria-label=\"Breadcrumb\"><ol class=\"bc\">'+items.map((item,i)=>{\n    const last=i===items.length-1;\n    return '<li'+(last?' aria-current=\"page\"':'')+'>'+(last?'<span>'+esc(item.label)+'</span>':'<a href=\"'+esc(item.href)+'\">'+esc(item.label)+'</a>')+'</li>';\n  }).join('')+'</ol></nav>';\n}\n/* .bc{display:flex;gap:0.25rem;list-style:none;padding:0} .bc li+li::before{content:'/';color:var(--text-dim);margin-right:0.25rem} */"},

    # ===== LAYOUT =====
    {"id": "layout-grid", "scaffolds": ["static", "dashboard", "form"], "keywords": ["layout", "grid", "holy grail", "sidebar", "header", "footer"], "always_for": [], "label": "CSS Grid holy grail layout",
     "code": "/* .layout{display:grid;grid-template-columns:240px 1fr;grid-template-rows:auto 1fr auto;min-height:100vh;grid-template-areas:'header header' 'sidebar main' 'footer footer'}\n.layout-header{grid-area:header} .layout-sidebar{grid-area:sidebar} .layout-main{grid-area:main} .layout-footer{grid-area:footer}\n@media(max-width:768px){.layout{grid-template-columns:1fr;grid-template-areas:'header' 'main' 'footer'} .layout-sidebar{display:none}} */"},
    {"id": "layout-masonry", "scaffolds": ["static"], "keywords": ["masonry", "pinterest", "waterfall", "columns"], "always_for": [], "label": "Masonry / Pinterest grid (CSS columns)",
     "code": "/* .masonry{column-count:3;column-gap:1rem;padding:1rem} .masonry-item{break-inside:avoid;margin-bottom:1rem;border-radius:var(--radius);overflow:hidden;background:var(--card)}\n@media(max-width:900px){.masonry{column-count:2}} @media(max-width:600px){.masonry{column-count:1}} */"},
    {"id": "layout-sticky", "scaffolds": ["static", "dashboard"], "keywords": ["sticky", "fixed", "header", "scroll-top", "pin"], "always_for": [], "label": "Sticky header + scroll-to-top",
     "code": "const header=document.querySelector('header'), scrollBtn=document.getElementById('scroll-top');\nwindow.addEventListener('scroll',()=>{header.classList.toggle('scrolled',scrollY>50);scrollBtn?.classList.toggle('visible',scrollY>300)});\nscrollBtn?.addEventListener('click',()=>window.scrollTo({top:0,behavior:'smooth'}));\n/* header{position:sticky;top:0;z-index:50;transition:padding 0.3s} header.scrolled{padding:0.5rem 1rem;box-shadow:0 2px 8px rgba(0,0,0,0.1)}\n#scroll-top{position:fixed;bottom:2rem;right:2rem;opacity:0;transition:opacity 0.3s} #scroll-top.visible{opacity:1} */"},
    {
        "id": "layout-splitpane",
        "scaffolds": ["static", "dashboard"],
        "keywords": ["split", "pane", "resize", "divider", "panel"],
        "always_for": [],
        "label": "Resizable split pane (drag divider between two panes)",
        "code": """\
// Resizable split pane. Drag the divider horizontally to resize the left
// pane (20-80% of the container). Listeners are on `document` so the drag
// keeps tracking even when the pointer leaves the divider.
function initSplit(id) {
  const container = document.getElementById(id);
  const leftPane = container.querySelector('.split-left');
  const divider = container.querySelector('.split-div');
  let dragging = false;

  divider.addEventListener('mousedown', () => { dragging = true; });

  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const rect = container.getBoundingClientRect();
    const pct = ((e.clientX - rect.left) / rect.width) * 100;
    leftPane.style.width = Math.max(20, Math.min(80, pct)) + '%';
  });

  document.addEventListener('mouseup', () => { dragging = false; });
}

/* Split-pane layout — a simple flexbox approach with a draggable divider.
.split       { display: flex; height: 100%; }
.split-left  { min-width: 20%; overflow: auto; }
.split-right { flex: 1; overflow: auto; }
.split-div   { width: 6px; background: var(--border); cursor: col-resize; }
.split-div:hover { background: var(--accent); }
*/""",
    },
    {"id": "layout-print", "scaffolds": ["static", "dashboard", "form"], "keywords": ["print", "pdf", "paper", "report", "invoice"], "always_for": [], "label": "Print-friendly stylesheet",
     "code": "/* @media print{body{background:#fff;color:#000;font-size:12pt} header,footer,nav,.sidebar,.no-print,button{display:none!important}\nmain{width:100%;margin:0;padding:0} a{color:#000;text-decoration:underline} a[href]::after{content:' ('attr(href)')';font-size:0.8em;color:#666}\ntable{border-collapse:collapse;width:100%} td,th{border:1px solid #ccc;padding:4px 8px} .page-break{page-break-before:always} @page{margin:2cm}} */"},

    # ===== DATA — Remaining =====
    {"id": "data-websocket", "scaffolds": ["static", "dashboard"], "keywords": ["websocket", "ws", "socket", "realtime", "push", "chat"], "always_for": [], "label": "WebSocket (auto-reconnect + message routing)",
     "code": "function createSocket(url, handlers) {\n  let ws, retries=0;\n  function connect() {\n    ws=new WebSocket(url);\n    ws.onopen=()=>{retries=0;handlers.onOpen?.()};\n    ws.onmessage=e=>{try{const m=JSON.parse(e.data);handlers[m.type]?.(m.data)}catch{}};\n    ws.onclose=()=>{if(retries<5){retries++;setTimeout(connect,1000*retries)}};\n  }\n  connect();\n  return{send:(type,data)=>{if(ws.readyState===1)ws.send(JSON.stringify({type,data}))},close:()=>ws.close()};\n}"},
    {"id": "data-router", "scaffolds": ["static"], "keywords": ["router", "route", "spa", "hash", "page", "navigate", "single page"], "always_for": [], "label": "Hash-based SPA router",
     "code": "function createRouter(routes, containerId) {\n  const el=document.getElementById(containerId);\n  function go(){const path=location.hash.slice(1)||'/';const route=routes[path]||routes['*']||(()=>'<h1>404</h1>');\n    el.innerHTML=typeof route==='function'?route():route;\n    document.querySelectorAll('a[href^=\"#\"]').forEach(a=>a.classList.toggle('active',a.getAttribute('href')==='#'+path));}\n  window.addEventListener('hashchange',go); go();\n  return{go:p=>{location.hash=p}};\n}"},

    # ===== ANIMATION — Remaining =====
    {"id": "anim-confetti", "scaffolds": ["static", "form", "game"], "keywords": ["confetti", "celebrate", "party", "win", "success"], "always_for": [], "label": "Confetti / celebration effect",
     "code": "function confetti(canvas, dur=2000) {\n  const ctx=canvas.getContext('2d'), ps=[], colors=['#f44','#4f4','#44f','#ff4','#f4f','#4ff'];\n  for(let i=0;i<100;i++) ps.push({x:canvas.width/2,y:canvas.height/2,vx:(Math.random()-0.5)*12,vy:Math.random()*-12-4,\n    color:colors[Math.floor(Math.random()*colors.length)],size:Math.random()*6+2,life:1});\n  const t0=Date.now();\n  function draw(){if(Date.now()-t0>dur)return; ctx.clearRect(0,0,canvas.width,canvas.height);\n    ps.forEach(p=>{p.x+=p.vx;p.y+=p.vy;p.vy+=0.2;p.life-=0.01;ctx.globalAlpha=Math.max(0,p.life);ctx.fillStyle=p.color;ctx.fillRect(p.x,p.y,p.size,p.size)});\n    ctx.globalAlpha=1;requestAnimationFrame(draw)}\n  draw();\n}"},
    {"id": "anim-typewriter", "scaffolds": ["static"], "keywords": ["typewriter", "typing", "text", "terminal", "console"], "always_for": [], "label": "Typewriter text effect",
     "code": "function typewrite(id, text, speed=40, onDone) {\n  const el=document.getElementById(id); el.textContent=''; let i=0;\n  function tick(){if(i<text.length){el.textContent+=text[i++];setTimeout(tick,speed)} else{el.classList.add('typed');if(onDone)onDone()}}\n  tick();\n}\n/* .typed::after{content:'|';animation:blink 1s step-end infinite} @keyframes blink{50%{opacity:0}} */"},
    {"id": "anim-canvas-draw", "scaffolds": ["static", "game"], "keywords": ["draw", "paint", "canvas", "sketch", "whiteboard", "brush"], "always_for": [], "label": "Canvas drawing / paint tool",
     "code": "function initDrawing(canvasId) {\n  const c=document.getElementById(canvasId), ctx=c.getContext('2d'); let drawing=false, lx=0, ly=0;\n  ctx.strokeStyle='#000'; ctx.lineWidth=3; ctx.lineCap='round';\n  function pos(e){const r=c.getBoundingClientRect(),t=e.touches?e.touches[0]:e;return[t.clientX-r.left,t.clientY-r.top]}\n  function start(e){drawing=true;[lx,ly]=pos(e)}\n  function draw(e){if(!drawing)return;e.preventDefault();const[x,y]=pos(e);ctx.beginPath();ctx.moveTo(lx,ly);ctx.lineTo(x,y);ctx.stroke();[lx,ly]=[x,y]}\n  c.addEventListener('mousedown',start);c.addEventListener('mousemove',draw);c.addEventListener('mouseup',()=>drawing=false);\n  c.addEventListener('touchstart',start);c.addEventListener('touchmove',draw);c.addEventListener('touchend',()=>drawing=false);\n  return{setColor:cl=>ctx.strokeStyle=cl,setSize:s=>ctx.lineWidth=s,clear:()=>ctx.clearRect(0,0,c.width,c.height)};\n}"},

    # ===== ACCESSIBILITY — Remaining =====
    {"id": "a11y-reduced-motion", "scaffolds": ["static", "form", "dashboard", "game"], "keywords": ["reduced", "motion", "prefers", "contrast", "accessibility"], "always_for": [], "label": "Reduced motion + high contrast respect",
     "code": "/* @media(prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:0.01ms!important;transition-duration:0.01ms!important;scroll-behavior:auto!important}}\n@media(prefers-contrast:high){:root{--bg:#000;--text:#fff;--accent:#ff0;--border:#fff} button,a{text-decoration:underline;border:2px solid currentColor}} */\nconst prefersReducedMotion = matchMedia('(prefers-reduced-motion:reduce)').matches;\n// Usage: if (!prefersReducedMotion) startAnimation();"},
    {
        "id": "a11y-keyboard",
        "scaffolds": ["static", "form", "dashboard", "game"],
        "keywords": ["keyboard", "shortcut", "hotkey", "keybind", "ctrl"],
        "always_for": [],
        "label": "Keyboard shortcuts manager (ignores typing in inputs)",
        "code": """\
// Keyboard shortcut registry. Skips keystrokes while the user is typing
// in an INPUT/TEXTAREA/contenteditable so Ctrl+A (select all) and friends
// still do the right thing in form fields.
const shortcuts = new Map();

function registerShortcut(key, description, handler) {
  shortcuts.set(key.toLowerCase(), { description, handler });
}

function shortcutKey(event) {
  const parts = [];
  if (event.ctrlKey || event.metaKey) parts.push('ctrl');
  if (event.shiftKey) parts.push('shift');
  if (event.altKey) parts.push('alt');
  parts.push(event.key.toLowerCase());
  return parts.join('+');
}

document.addEventListener('keydown', (e) => {
  const el = e.target;
  if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) {
    return;
  }
  const match = shortcuts.get(shortcutKey(e));
  if (match) {
    e.preventDefault();
    match.handler();
  }
});

// Usage:
//   registerShortcut('ctrl+s',       'Save',      () => save());
//   registerShortcut('ctrl+shift+z', 'Redo',      () => redo());
//   registerShortcut('?',            'Show help', () => openHelp());""",
    },

    # ===== MEDIA — Remaining =====
    {"id": "media-video", "scaffolds": ["static"], "keywords": ["video", "player", "stream", "embed", "youtube", "media"], "always_for": [], "label": "Video player controls (custom)",
     "code": "function initVideo(videoId, playBtnId, progressId) {\n  const v=document.getElementById(videoId), btn=document.getElementById(playBtnId), prog=document.getElementById(progressId);\n  btn.onclick=()=>{if(v.paused){v.play();btn.textContent='\\u23F8'}else{v.pause();btn.textContent='\\u25B6'}};\n  v.ontimeupdate=()=>{prog.value=(v.currentTime/v.duration)*100||0};\n  prog.oninput=()=>{v.currentTime=(prog.value/100)*v.duration};\n  v.onended=()=>{btn.textContent='\\u25B6';prog.value=0};\n}"},
    {"id": "media-map", "scaffolds": ["static", "dashboard"], "keywords": ["map", "geo", "location", "leaflet", "coordinate", "marker", "gps"], "always_for": [], "label": "Interactive map (Leaflet pattern)",
     "code": "// Leaflet map pattern (CDN: unpkg.com/leaflet@1.9/dist/leaflet.{js,css})\nfunction initMap(containerId, lat=51.505, lng=-0.09, zoom=13) {\n  const map = L.map(containerId).setView([lat, lng], zoom);\n  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom:19, attribution:'OpenStreetMap' }).addTo(map);\n  return {\n    addMarker: (lat,lng,popup) => { const m = L.marker([lat,lng]).addTo(map); if(popup) m.bindPopup(popup); return m; },\n    fitBounds: (markers) => { if(markers.length) map.fitBounds(markers.map(m=>[m.getLatLng().lat,m.getLatLng().lng])); }\n  };\n}"},
    # ===== Final 7 — filling remaining gaps =====
    {"id": "form-sortable", "scaffolds": ["form", "static"], "keywords": ["sort", "sortable", "reorder", "priority", "rank", "order", "arrange"], "always_for": [], "label": "Sortable list (drag to reorder)",
     "code": "function initSortable(listId) {\n  const list=document.getElementById(listId); let dragged=null;\n  list.querySelectorAll('.sortable-item').forEach(item=>{\n    item.draggable=true;\n    item.ondragstart=e=>{dragged=item;item.classList.add('dragging');e.dataTransfer.effectAllowed='move'};\n    item.ondragend=()=>{dragged=null;item.classList.remove('dragging')};\n    item.ondragover=e=>{e.preventDefault();if(item!==dragged){const rect=item.getBoundingClientRect();\n      const mid=rect.top+rect.height/2;if(e.clientY<mid)list.insertBefore(dragged,item);else list.insertBefore(dragged,item.nextSibling)}};\n  });\n}\n/* .dragging{opacity:0.4} .sortable-item{padding:0.75rem;border:1px solid var(--border);border-radius:var(--radius);cursor:grab;margin-bottom:4px} */"},
    {
        "id": "data-local-db",
        "scaffolds": ["static", "form"],
        "keywords": ["indexeddb", "database", "offline", "cache", "store", "persist", "idb"],
        "always_for": [],
        "label": "IndexedDB wrapper (offline-first storage, promise-based)",
        "code": """\
// Promise-based IndexedDB wrapper. Use this for anything bigger than
// a few KB or when you need queries — localStorage maxes out around 5MB
// and blocks the main thread on writes.

function openDB(name, version, storeName) {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(name, version);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(storeName, { keyPath: 'id', autoIncrement: true });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function dbPut(db, store, item) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readwrite');
    tx.objectStore(store).put(item);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

function dbGetAll(db, store) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store);
    const req = tx.objectStore(store).getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function dbDelete(db, store, key) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readwrite');
    tx.objectStore(store).delete(key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

// Usage:
//   const db = await openDB('myapp', 1, 'items');
//   await dbPut(db, 'items', { title: 'Hello' });
//   const rows = await dbGetAll(db, 'items');""",
    },
    {"id": "ui-drawer", "scaffolds": ["static", "form", "dashboard"], "keywords": ["drawer", "panel", "slide", "off-canvas", "sidebar", "mobile"], "always_for": [], "label": "Slide-in drawer / off-canvas panel",
     "code": "function initDrawer(toggleId, drawerId) {\n  const toggle=document.getElementById(toggleId), drawer=document.getElementById(drawerId);\n  const backdrop=document.createElement('div'); backdrop.className='drawer-backdrop hidden'; document.body.appendChild(backdrop);\n  function open(){drawer.classList.add('open');backdrop.classList.remove('hidden');document.body.style.overflow='hidden'}\n  function close(){drawer.classList.remove('open');backdrop.classList.add('hidden');document.body.style.overflow=''}\n  toggle.onclick=()=>drawer.classList.contains('open')?close():open();\n  backdrop.onclick=close;\n  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&drawer.classList.contains('open'))close()});\n}\n/* .drawer{position:fixed;top:0;left:-280px;width:280px;height:100%;background:var(--card);transition:left 0.3s;z-index:100} .drawer.open{left:0}\n.drawer-backdrop{position:fixed;inset:0;background:rgba(0,0,0,0.4);z-index:99} */"},
    {"id": "media-webcam", "scaffolds": ["static", "form"], "keywords": ["webcam", "camera", "video", "capture", "photo", "stream", "selfie"], "always_for": [], "label": "Webcam capture (getUserMedia + snapshot)",
     "code": "async function initWebcam(videoId, canvasId, captureBtnId) {\n  const video=document.getElementById(videoId), canvas=document.getElementById(canvasId), ctx=canvas.getContext('2d');\n  try { const stream = await navigator.mediaDevices.getUserMedia({video:true}); video.srcObject=stream; video.play(); }\n  catch(e) { console.error('Camera access denied:', e); return; }\n  document.getElementById(captureBtnId).onclick=()=>{\n    canvas.width=video.videoWidth; canvas.height=video.videoHeight;\n    ctx.drawImage(video,0,0); canvas.classList.remove('hidden');\n    // Get as data URL: canvas.toDataURL('image/png')\n  };\n}"},
    {
        "id": "anim-number-counter",
        "scaffolds": ["static", "dashboard"],
        "keywords": ["counter", "count", "number", "animate", "increment", "stat", "kpi"],
        "always_for": [],
        "label": "Animated number counter (ease-out count-up effect)",
        "code": """\
// Animated count-up — smoothly tweens a number from its current textContent
// value to a target over `duration` ms. Uses an ease-out cubic curve so the
// animation lands at the target instead of overshooting.
function animateCount(elementId, target, duration = 1000, prefix = '', suffix = '') {
  const el = document.getElementById(elementId);
  const start = parseInt(el.textContent.replace(/[^0-9-]/g, ''), 10) || 0;
  const range = target - start;
  const t0 = performance.now();

  function step(now) {
    const progress = Math.min(1, (now - t0) / duration);
    const eased = 1 - Math.pow(1 - progress, 3);          // ease-out cubic
    const value = Math.round(start + range * eased);
    el.textContent = prefix + value.toLocaleString() + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }

  requestAnimationFrame(step);
}

// Usage: animateCount('revenue', 45230, 1500, '$');
//        animateCount('pct', 92, 900, '', '%');""",
    },
    {"id": "data-form-serialize", "scaffolds": ["form"], "keywords": ["serialize", "formdata", "collect", "extract", "json", "submit"], "always_for": [], "label": "Form serialization to JSON",
     "code": "// Collect all form inputs into a clean JSON object\nfunction serializeForm(formId) {\n  const form=document.getElementById(formId), data={};\n  new FormData(form).forEach((value, key) => {\n    if (data[key]) { // Handle multiple values (checkboxes, multi-select)\n      if (!Array.isArray(data[key])) data[key] = [data[key]];\n      data[key].push(value);\n    } else { data[key] = value; }\n  });\n  // Convert numeric strings\n  for (const k in data) { if (typeof data[k]==='string' && data[k] && !isNaN(data[k])) data[k] = +data[k]; }\n  return data;\n}\n// Usage: form.onsubmit = (e) => { e.preventDefault(); const data = serializeForm('my-form'); console.log(data); };"},
    {"id": "static-chat", "scaffolds": ["static"], "keywords": ["chat", "message", "conversation", "messenger", "bubble", "send", "inbox"], "always_for": [], "label": "Chat / messenger interface",
     "code": "function initChat(containerId, inputId, sendBtnId) {\n  const msgs=document.getElementById(containerId), input=document.getElementById(inputId);\n  const esc=s=>s.replace(/[<>&\"]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;','\"':'&quot;'})[c]);\n  function addMsg(text, from='user') {\n    const el=document.createElement('div'); el.className='msg msg-'+from;\n    el.innerHTML='<div class=\"msg-bubble\">'+esc(text)+'</div><span class=\"msg-time\">'+new Date().toLocaleTimeString()+'</span>';\n    msgs.appendChild(el); msgs.scrollTop=msgs.scrollHeight;\n  }\n  document.getElementById(sendBtnId).onclick=()=>{if(!input.value.trim())return;addMsg(input.value.trim());input.value='';input.focus()};\n  input.onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById(sendBtnId).click()}};\n}\n/* .msg{display:flex;flex-direction:column;margin:0.5rem 0} .msg-user{align-items:flex-end} .msg-bubble{max-width:70%;padding:0.5rem 1rem;border-radius:12px;background:var(--card)} .msg-user .msg-bubble{background:var(--accent);color:#fff} */"},

    # ===== COMPOSITE REFS (multi-pattern wiring — bench-gap coverage) =====
    # These show how to combine 2-4 primitives into a complete feature.
    # Small models pattern-match on these more reliably than on three
    # separate snippets they have to integrate themselves.
    {
        "id": "composite-pomodoro",
        "scaffolds": ["static", "form"],
        "keywords": ["pomodoro", "focus", "session", "work break", "25 minute", "timer cycle", "phase"],
        "always_for": [],
        "label": "Pomodoro timer (SVG progress ring + phase switch + session count + localStorage)",
        "code": """\
// Pomodoro — work/break phases with circular SVG progress ring.
// Shows: stroke-dasharray math, phase state machine, session persistence,
// start/pause/reset tri-state, and an audio alert at phase end.
(function () {
  'use strict';

  const WORK_SECONDS = 25 * 60;
  const BREAK_SECONDS = 5 * 60;
  const STORAGE_KEY = 'pomodoro.sessions';

  // SVG ring: circumference = 2πr. Animate stroke-dashoffset from C to 0
  // as progress goes 0 → 1 so the ring "fills" over the phase.
  const ring = document.getElementById('ring-progress');
  const radius = Number(ring.getAttribute('r')) || 54;
  const CIRC = 2 * Math.PI * radius;
  ring.style.strokeDasharray = String(CIRC);
  ring.style.strokeDashoffset = String(CIRC);

  const displayEl = document.getElementById('time-display');
  const phaseEl = document.getElementById('phase-label');
  const countEl = document.getElementById('session-count');
  const btnStart = document.getElementById('btn-start');
  const btnPause = document.getElementById('btn-pause');
  const btnReset = document.getElementById('btn-reset');
  const alertAudio = document.getElementById('alert-sound');

  const state = {
    phase: 'work',        // 'work' | 'break'
    remaining: WORK_SECONDS,
    running: false,
    tickId: null,
    sessions: Number(localStorage.getItem(STORAGE_KEY) || 0),
  };

  function format(seconds) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
  }

  function render() {
    displayEl.textContent = format(state.remaining);
    phaseEl.textContent = state.phase === 'work' ? 'Focus' : 'Break';
    countEl.textContent = String(state.sessions);

    const total = state.phase === 'work' ? WORK_SECONDS : BREAK_SECONDS;
    const progress = 1 - (state.remaining / total);     // 0 → 1
    ring.style.strokeDashoffset = String(CIRC * (1 - progress));
  }

  function tick() {
    state.remaining -= 1;
    if (state.remaining <= 0) {
      advancePhase();
    }
    render();
  }

  function advancePhase() {
    // Alert + phase flip. Count sessions only at the END of a work block.
    try { alertAudio.currentTime = 0; alertAudio.play().catch(() => {}); } catch {}
    if (state.phase === 'work') {
      state.sessions += 1;
      localStorage.setItem(STORAGE_KEY, String(state.sessions));
      state.phase = 'break';
      state.remaining = BREAK_SECONDS;
    } else {
      state.phase = 'work';
      state.remaining = WORK_SECONDS;
    }
  }

  function start() {
    if (state.running) return;
    state.running = true;
    state.tickId = setInterval(tick, 1000);
  }

  function pause() {
    state.running = false;
    if (state.tickId) { clearInterval(state.tickId); state.tickId = null; }
  }

  function reset() {
    pause();
    state.phase = 'work';
    state.remaining = WORK_SECONDS;
    render();
  }

  btnStart.addEventListener('click', start);
  btnPause.addEventListener('click', pause);
  btnReset.addEventListener('click', reset);
  render();
})();""",
    },
    {
        "id": "composite-palette-math",
        "scaffolds": ["form", "static"],
        "keywords": ["palette", "color scheme", "analogous", "complementary", "triadic", "tetradic", "hsl", "hex"],
        "always_for": [],
        "label": "Color palette math (HSL conversion + scheme generation + copy-hex)",
        "code": """\
// Color palette math — hex ↔ HSL and the four classic schemes.
// Shows: conversion in both directions, hue-rotation helpers,
// scheme generators, and a copy-to-clipboard pattern for each swatch.

function hexToHsl(hex) {
  const clean = hex.replace('#', '');
  const r = parseInt(clean.slice(0, 2), 16) / 255;
  const g = parseInt(clean.slice(2, 4), 16) / 255;
  const b = parseInt(clean.slice(4, 6), 16) / 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const l = (max + min) / 2;
  let h = 0, s = 0;
  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r: h = (g - b) / d + (g < b ? 6 : 0); break;
      case g: h = (b - r) / d + 2; break;
      case b: h = (r - g) / d + 4; break;
    }
    h *= 60;
  }
  return { h: Math.round(h), s: Math.round(s * 100), l: Math.round(l * 100) };
}

function hslToHex(h, s, l) {
  s /= 100; l /= 100;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let r = 0, g = 0, b = 0;
  if (h < 60)       { r = c; g = x; }
  else if (h < 120) { r = x; g = c; }
  else if (h < 180) {         g = c; b = x; }
  else if (h < 240) {         g = x; b = c; }
  else if (h < 300) { r = x;         b = c; }
  else              { r = c;         b = x; }
  const toHex = (v) => Math.round((v + m) * 255).toString(16).padStart(2, '0');
  return '#' + toHex(r) + toHex(g) + toHex(b);
}

function rotateHue(hex, degrees) {
  const hsl = hexToHsl(hex);
  return hslToHex((hsl.h + degrees + 360) % 360, hsl.s, hsl.l);
}

// The four classic schemes — each returns the base + derivatives.
function schemeAnalogous(hex)     { return [rotateHue(hex, -30), hex, rotateHue(hex, 30)]; }
function schemeComplementary(hex) { return [hex, rotateHue(hex, 180)]; }
function schemeTriadic(hex)       { return [hex, rotateHue(hex, 120), rotateHue(hex, 240)]; }
function schemeTetradic(hex)      { return [hex, rotateHue(hex, 90), rotateHue(hex, 180), rotateHue(hex, 270)]; }

// Copy-to-clipboard wiring with a brief "Copied!" affordance on the button.
async function copyHex(hex, button) {
  try {
    await navigator.clipboard.writeText(hex);
    const original = button.textContent;
    button.textContent = 'Copied!';
    setTimeout(() => { button.textContent = original; }, 1200);
  } catch {
    // Older browsers — fall back to the deprecated execCommand path.
    const ta = document.createElement('textarea');
    ta.value = hex; document.body.appendChild(ta);
    ta.select(); document.execCommand('copy'); ta.remove();
  }
}

// Usage: given an input#base (color) and a grid container per scheme,
// re-render swatches on input change, each with a copy button.
function renderSwatches(containerId, colors) {
  const esc = (s) => s.replace(/[<>&"]/g, (c) => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'}[c]));
  const container = document.getElementById(containerId);
  container.innerHTML = colors.map((hex) =>
    '<button class="swatch" style="background:' + esc(hex) + '" data-hex="' + esc(hex) + '">' +
    '<span class="swatch-hex">' + esc(hex) + '</span></button>'
  ).join('');
  container.querySelectorAll('.swatch').forEach((btn) => {
    btn.addEventListener('click', () => copyHex(btn.dataset.hex, btn));
  });
}""",
    },
    {
        "id": "composite-markdown-splitpane",
        "scaffolds": ["static"],
        "keywords": ["markdown editor", "live preview", "split pane", "notes editor", "side by side"],
        "always_for": [],
        "label": "Markdown editor with split-pane live preview + multi-note sidebar + search",
        "code": """\
// Markdown editor — split pane (editor ↔ preview), multi-note sidebar,
// search filter, and localStorage persistence keyed by note id.
(function () {
  'use strict';

  const STORAGE_KEY = 'md.notes';
  const loadNotes = () => JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
  const saveNotes = () => localStorage.setItem(STORAGE_KEY, JSON.stringify(state.notes));

  const state = {
    notes: loadNotes(),
    activeId: null,
    filter: '',
  };

  if (!state.notes.length) {
    state.notes.push({ id: Date.now(), title: 'Welcome', body: '# Welcome\\nStart typing...' });
  }
  state.activeId = state.notes[0].id;

  const esc = (s) => s.replace(/[<>&"]/g, (c) => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'}[c]));

  // Minimal markdown → HTML (headings, bold, italic, code, lists, paragraphs).
  // Escape FIRST, then apply substitutions — prevents XSS via user markdown.
  function renderMarkdown(md) {
    return esc(md)
      .replace(/^### (.+)$/gm, '<h3>$1</h3>')
      .replace(/^## (.+)$/gm, '<h2>$1</h2>')
      .replace(/^# (.+)$/gm, '<h1>$1</h1>')
      .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
      .replace(/\\*(.+?)\\*/g, '<em>$1</em>')
      .replace(/`([^`]+?)`/g, '<code>$1</code>')
      .replace(/^- (.+)$/gm, '<li>$1</li>')
      .replace(/(<li>[\\s\\S]*?<\\/li>)/g, '<ul>$1</ul>')
      .replace(/\\n\\n/g, '</p><p>')
      .replace(/\\n/g, '<br>');
  }

  const sidebar = document.getElementById('note-list');
  const searchInput = document.getElementById('note-search');
  const editor = document.getElementById('editor');
  const preview = document.getElementById('preview');
  const titleInput = document.getElementById('note-title');
  const btnNew = document.getElementById('btn-new-note');

  function visibleNotes() {
    const q = state.filter.toLowerCase();
    if (!q) return state.notes;
    return state.notes.filter((n) =>
      n.title.toLowerCase().includes(q) || n.body.toLowerCase().includes(q)
    );
  }

  function renderSidebar() {
    sidebar.innerHTML = visibleNotes().map((n) =>
      '<button class="note-item' + (n.id === state.activeId ? ' active' : '') +
      '" data-id="' + n.id + '">' + esc(n.title || 'Untitled') + '</button>'
    ).join('');
    sidebar.querySelectorAll('.note-item').forEach((el) => {
      el.addEventListener('click', () => selectNote(Number(el.dataset.id)));
    });
  }

  function renderEditor() {
    const note = state.notes.find((n) => n.id === state.activeId);
    if (!note) return;
    titleInput.value = note.title;
    editor.value = note.body;
    preview.innerHTML = '<p>' + renderMarkdown(note.body) + '</p>';
  }

  function selectNote(id) {
    state.activeId = id;
    renderSidebar();
    renderEditor();
  }

  function updateActive(patch) {
    const note = state.notes.find((n) => n.id === state.activeId);
    if (!note) return;
    Object.assign(note, patch);
    saveNotes();
  }

  editor.addEventListener('input', () => {
    updateActive({ body: editor.value });
    preview.innerHTML = '<p>' + renderMarkdown(editor.value) + '</p>';
  });
  titleInput.addEventListener('input', () => {
    updateActive({ title: titleInput.value });
    renderSidebar();
  });
  searchInput.addEventListener('input', () => {
    state.filter = searchInput.value;
    renderSidebar();
  });
  btnNew.addEventListener('click', () => {
    const note = { id: Date.now(), title: 'New note', body: '' };
    state.notes.unshift(note);
    state.activeId = note.id;
    saveNotes();
    renderSidebar();
    renderEditor();
    titleInput.focus(); titleInput.select();
  });

  renderSidebar();
  renderEditor();
})();

/* Split-pane layout — one CSS grid does the work. Sidebar + editor + preview.
.app-shell { display: grid; grid-template-columns: 240px 1fr 1fr; height: 100vh; }
.app-shell > aside { border-right: 1px solid var(--border); overflow-y: auto; }
.app-shell > .editor-pane { border-right: 1px solid var(--border); }
.app-shell > .preview-pane { padding: var(--space-lg); overflow-y: auto; }
#editor { width: 100%; height: 100%; border: 0; padding: var(--space-lg); resize: none; font-family: var(--font-mono); }
@media (max-width: 900px) { .app-shell { grid-template-columns: 1fr; grid-template-rows: auto 1fr 1fr; } }
*/""",
    },
    {
        "id": "composite-kanban",
        "scaffolds": ["static"],
        "keywords": ["kanban", "board", "columns", "cards between columns", "drag cards", "undo redo"],
        "always_for": [],
        "label": "Kanban board composite (columns + drag-drop + card editor + undo/redo + persistence)",
        "code": """\
// Kanban composite — multi-column drag/drop, card edit modal, undo/redo
// stack synced to localStorage. Every mutation goes through applyOp() so
// the history stack is correct regardless of which control triggered it.
(function () {
  'use strict';

  const STORAGE_KEY = 'kanban.state';
  const esc = (s) => s.replace(/[<>&"]/g, (c) => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'}[c]));

  const defaultState = () => ({
    columns: [
      { id: 'todo', name: 'To Do', cards: [{ id: 'c1', text: 'Example card' }] },
      { id: 'doing', name: 'Doing', cards: [] },
      { id: 'done', name: 'Done', cards: [] },
    ],
  });

  const loadState = () => {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || defaultState(); }
    catch { return defaultState(); }
  };

  // Undo stack: snapshot BEFORE every mutation; redo stack: populated on undo.
  const history = { past: [], future: [] };
  let state = loadState();

  function snapshot() {
    history.past.push(JSON.stringify(state));
    history.future.length = 0;          // New edit invalidates redo.
    if (history.past.length > 50) history.past.shift();
  }

  function applyOp(fn) {
    snapshot();
    fn(state);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    render();
  }

  function undo() {
    if (!history.past.length) return;
    history.future.push(JSON.stringify(state));
    state = JSON.parse(history.past.pop());
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    render();
  }

  function redo() {
    if (!history.future.length) return;
    history.past.push(JSON.stringify(state));
    state = JSON.parse(history.future.pop());
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    render();
  }

  const boardEl = document.getElementById('board');

  function render() {
    boardEl.innerHTML = state.columns.map((col) =>
      '<section class="column" data-col="' + esc(col.id) + '">' +
        '<header>' + esc(col.name) + '</header>' +
        '<div class="cards">' +
          col.cards.map((card) =>
            '<article class="card" draggable="true" data-card="' + esc(card.id) +
            '">' + esc(card.text) + '</article>'
          ).join('') +
        '</div>' +
        '<button class="add-card" data-col="' + esc(col.id) + '">+ Add card</button>' +
      '</section>'
    ).join('');
    wireEvents();
  }

  function findCard(cardId) {
    for (const col of state.columns) {
      const idx = col.cards.findIndex((c) => c.id === cardId);
      if (idx >= 0) return { column: col, index: idx, card: col.cards[idx] };
    }
    return null;
  }

  function wireEvents() {
    // Drag state is tracked outside the element so dragover can consult it.
    let draggedCardId = null;

    boardEl.querySelectorAll('.card').forEach((el) => {
      el.addEventListener('dragstart', (e) => {
        draggedCardId = el.dataset.card;
        e.dataTransfer.effectAllowed = 'move';
        el.classList.add('dragging');
      });
      el.addEventListener('dragend', () => { el.classList.remove('dragging'); });
      el.addEventListener('dblclick', () => {
        const hit = findCard(el.dataset.card);
        if (!hit) return;
        const next = prompt('Edit card', hit.card.text);
        if (next !== null) {
          applyOp(() => { hit.card.text = next; });
        }
      });
    });

    boardEl.querySelectorAll('.column').forEach((col) => {
      col.addEventListener('dragover', (e) => { e.preventDefault(); col.classList.add('drop-target'); });
      col.addEventListener('dragleave', () => col.classList.remove('drop-target'));
      col.addEventListener('drop', (e) => {
        e.preventDefault();
        col.classList.remove('drop-target');
        if (!draggedCardId) return;
        const targetColId = col.dataset.col;
        applyOp((s) => {
          const hit = findCard(draggedCardId);
          if (!hit) return;
          hit.column.cards.splice(hit.index, 1);
          const target = s.columns.find((c) => c.id === targetColId);
          if (target) target.cards.push(hit.card);
        });
        draggedCardId = null;
      });
    });

    boardEl.querySelectorAll('.add-card').forEach((btn) => {
      btn.addEventListener('click', () => {
        const text = prompt('New card text');
        if (!text) return;
        applyOp((s) => {
          const target = s.columns.find((c) => c.id === btn.dataset.col);
          if (target) target.cards.push({ id: 'c' + Date.now(), text });
        });
      });
    });
  }

  // Keyboard shortcuts — Ctrl+Z / Ctrl+Shift+Z. Matches the Mac default
  // (Cmd+Z / Cmd+Shift+Z) because we check metaKey too.
  document.addEventListener('keydown', (e) => {
    const mod = e.ctrlKey || e.metaKey;
    if (!mod) return;
    if (e.key.toLowerCase() === 'z') {
      e.preventDefault();
      if (e.shiftKey) redo(); else undo();
    }
  });

  render();
})();""",
    },
    {
        "id": "composite-micro-complete",
        "scaffolds": ["static", "form"],
        "keywords": ["simple", "minimal", "starter", "complete example", "tiny app", "small app"],
        "always_for": [],
        "label": "Complete minimal app (HTML + CSS + JS, under ~100 lines total, production-ready shape)",
        "code": """\
<!-- index.html — a complete, production-shaped minimal app. Small models
     should treat this as the baseline for "what 3 files wired together
     look like" rather than reaching for a framework. -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Click Counter</title>
</head>
<body>
  <main id="app">
    <h1>Clicks</h1>
    <div id="count" class="count" aria-live="polite">0</div>
    <div class="actions">
      <button id="btn-inc" class="btn btn-primary">+1</button>
      <button id="btn-reset" class="btn btn-secondary">Reset</button>
    </div>
  </main>
</body>
</html>

/* styles.css — inherit the design-system tokens from static-base.
   Keep the page-level styles short: layout, not theme. */
body { display: flex; align-items: center; justify-content: center;
       min-height: 100vh; padding: var(--space-lg); }
#app { text-align: center; }
h1   { font-size: var(--text-xl); color: var(--text-dim);
       text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: var(--space-md); }
.count { font-size: 6rem; font-weight: 700; color: var(--accent);
         line-height: 1; margin-bottom: var(--space-xl); }
.actions { display: flex; gap: var(--space-md); justify-content: center; }

// app.js — one IIFE, one state object, one render call per mutation.
(function () {
  'use strict';

  const STORAGE_KEY = 'counter.value';
  const state = { count: Number(localStorage.getItem(STORAGE_KEY) || 0) };

  const countEl = document.getElementById('count');
  const btnInc = document.getElementById('btn-inc');
  const btnReset = document.getElementById('btn-reset');

  function render() {
    countEl.textContent = String(state.count);
    localStorage.setItem(STORAGE_KEY, String(state.count));
  }

  btnInc.addEventListener('click', () => { state.count += 1; render(); });
  btnReset.addEventListener('click', () => { state.count = 0; render(); });

  render();
})();""",
    },

    # ===== VISUAL DESIGN STYLES (cross-scaffold, keyword-matched) =====
    {
        "id": "style-brutalist",
        "scaffolds": ["static", "form", "dashboard"],
        "keywords": ["brutalist", "editorial", "newspaper", "bold", "stark", "swiss", "typography", "monochrome", "sharp", "blocky", "industrial", "black and white"],
        "always_for": [],
        "label": "Brutalist / editorial design (bold borders, inverted hovers, no decoration)",
        "code": """\
/* Brutalist: sharp corners, thick borders, aggressive hover states, newspaper columns */
:root{--bg:#fff;--fg:#000;--accent:#000;--border:3px solid #000;--font-display:"Helvetica Neue",Arial,sans-serif;--font-body:Georgia,serif;--space:2rem}
body{background:var(--bg);color:var(--fg);font-family:var(--font-body);line-height:1.5}
h1,h2,h3{font-family:var(--font-display);text-transform:uppercase;letter-spacing:-.02em;font-weight:900}
h1{font-size:clamp(3rem,8vw,6rem);line-height:1;border-bottom:var(--border);padding-bottom:var(--space)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:0;border-top:var(--border)}
.card{border-bottom:var(--border);border-right:var(--border);padding:var(--space);transition:background .1s,color .1s}
.card:hover{background:#000;color:#fff}
nav a{padding:1rem var(--space);text-decoration:none;color:var(--fg);text-transform:uppercase;font-weight:700;font-size:.85rem;
  letter-spacing:.1em;border-right:var(--border);transition:background .1s,color .1s}
nav a:hover{background:var(--fg);color:var(--bg)}
button{background:var(--fg);color:var(--bg);border:none;padding:.8rem 2rem;text-transform:uppercase;font-weight:700;letter-spacing:.1em;cursor:pointer}
button:hover{background:#333}
input{border:var(--border);padding:.8rem;font-family:var(--font-body);font-size:1rem;width:100%;background:var(--bg)}
input:focus{outline:none;box-shadow:4px 4px 0 var(--fg)}""",
    },
    {
        "id": "style-glass",
        "scaffolds": ["static", "form", "dashboard"],
        "keywords": ["glass", "glassmorphism", "frosted", "translucent", "blur", "gradient", "glow", "ios", "floating", "futuristic", "sleek"],
        "always_for": [],
        "label": "Glassmorphism / frosted (backdrop-blur, translucent cards, gradient glow)",
        "code": """\
/* Glassmorphism: translucent cards, backdrop blur, soft glow, gradient bg */
:root{--bg-start:#1a0533;--bg-end:#0a1628;--glass:rgba(255,255,255,.08);--glass-border:rgba(255,255,255,.15);
  --glass-hover:rgba(255,255,255,.12);--text:#f0eef6;--text-dim:rgba(240,238,246,.6);
  --accent:#a78bfa;--accent-glow:rgba(167,139,250,.4);--radius:16px;--blur:20px;--font:"Inter",system-ui,sans-serif}
body{min-height:100vh;background:linear-gradient(135deg,var(--bg-start),var(--bg-end));color:var(--text);font-family:var(--font)}
body::before{content:"";position:fixed;inset:0;background:radial-gradient(circle at 30% 40%,rgba(120,80,220,.15),transparent 50%),
  radial-gradient(circle at 70% 60%,rgba(60,120,220,.1),transparent 50%);pointer-events:none}
h1{background:linear-gradient(135deg,#fff,var(--accent));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.glass{background:var(--glass);backdrop-filter:blur(var(--blur));-webkit-backdrop-filter:blur(var(--blur));
  border:1px solid var(--glass-border);border-radius:var(--radius);padding:1.5rem;transition:background .3s,box-shadow .3s}
.glass:hover{background:var(--glass-hover);box-shadow:0 8px 32px rgba(0,0,0,.3),inset 0 0 0 1px rgba(255,255,255,.1)}
button{background:linear-gradient(135deg,var(--accent),#7c3aed);color:#fff;border:none;padding:.75rem 1.8rem;
  border-radius:50px;font-weight:600;cursor:pointer;transition:box-shadow .3s,transform .2s}
button:hover{box-shadow:0 4px 24px var(--accent-glow);transform:translateY(-1px)}
input{background:rgba(255,255,255,.06);border:1px solid var(--glass-border);border-radius:12px;padding:.75rem 1rem;
  color:var(--text);transition:border-color .2s}
input:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-glow);outline:none}""",
    },
    {
        "id": "style-terminal",
        "scaffolds": ["static", "form", "dashboard"],
        "keywords": ["terminal", "hacker", "retro", "crt", "matrix", "green", "console", "command", "cli", "coding", "developer"],
        "always_for": [],
        "label": "Retro terminal / hacker (green-on-black, scanlines, CRT glow)",
        "code": """\
/* Retro terminal: monospace, green phosphor, scanlines, CRT curvature */
:root{--bg:#0d0d0d;--fg:#00ff41;--fg-dim:#00aa2a;--amber:#ffb000;--glow:0 0 8px rgba(0,255,65,.6);
  --border-glow:0 0 2px var(--fg),inset 0 0 2px var(--fg);--font:"Courier New",monospace}
body{background:var(--bg);color:var(--fg);font-family:var(--font);font-size:14px;line-height:1.7}
body::before{content:"";position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.15) 2px,rgba(0,0,0,.15) 4px);pointer-events:none;z-index:1000}
body::after{content:"";position:fixed;inset:0;background:radial-gradient(ellipse at center,transparent 60%,rgba(0,0,0,.6));pointer-events:none;z-index:999}
h1{font-weight:normal;text-transform:uppercase;letter-spacing:.15em;text-shadow:var(--glow)}
h1::before{content:"> ";color:var(--amber)}
.panel{border:1px solid var(--fg-dim);padding:1.2rem;box-shadow:var(--border-glow)}
.cursor{display:inline-block;width:.6em;height:1.1em;background:var(--fg);animation:blink 1s step-end infinite}
@keyframes blink{50%{opacity:0}}
button{background:transparent;color:var(--fg);border:1px solid var(--fg);padding:.5rem 1.5rem;text-transform:uppercase;
  letter-spacing:.1em;cursor:pointer;transition:all .15s}
button:hover{background:var(--fg);color:var(--bg);box-shadow:0 0 12px rgba(0,255,65,.5)}
input{background:rgba(0,255,65,.05);border:1px solid var(--fg-dim);padding:.5rem;color:var(--fg);font-family:var(--font)}
input:focus{border-color:var(--fg);box-shadow:var(--border-glow);outline:none}
::selection{background:var(--fg);color:var(--bg)}""",
    },
    {
        "id": "style-scandinavian",
        "scaffolds": ["static", "form", "dashboard"],
        "keywords": ["minimal", "scandinavian", "warm", "clean", "elegant", "muji", "kinfolk", "organic", "natural", "earthy", "calm", "zen", "simple", "quiet"],
        "always_for": [],
        "label": "Warm minimal / Scandinavian (serif headings, earth tones, generous whitespace)",
        "code": """\
/* Warm minimal: off-white, earth tones, serif headings, generous breathing room */
:root{--bg:#f5f0eb;--bg-card:#fff;--fg:#2c2825;--fg-dim:#8a8078;--accent:#9a7b6b;--border:#e0d8cf;
  --font-serif:Georgia,"Times New Roman",serif;--font-sans:"Helvetica Neue",Arial,sans-serif;
  --space:2rem;--radius:4px;--transition:300ms ease}
body{background:var(--bg);color:var(--fg);font-family:var(--font-sans);font-size:16px;line-height:1.75;-webkit-font-smoothing:antialiased}
h1,h2,h3{font-family:var(--font-serif);font-weight:400;letter-spacing:-.01em}
h1{font-size:clamp(2rem,4vw,3rem);line-height:1.2}
.card{background:var(--bg-card);border:1px solid rgba(224,216,207,.5);border-radius:var(--radius);padding:calc(var(--space)*1.25);
  transition:border-color var(--transition)}
.card:hover{border-color:var(--border)}
nav a{color:var(--fg-dim);text-decoration:none;font-size:.9rem;position:relative;transition:color var(--transition)}
nav a::after{content:"";position:absolute;bottom:-2px;left:0;width:0;height:1px;background:var(--fg);transition:width var(--transition)}
nav a:hover{color:var(--fg)} nav a:hover::after{width:100%}
button{background:var(--fg);color:var(--bg);border:none;padding:.7rem 1.8rem;border-radius:var(--radius);
  font-size:.9rem;cursor:pointer;transition:background var(--transition)}
button:hover{background:var(--accent)}
.btn-outline{background:transparent;color:var(--fg);border:1px solid var(--border)}
.btn-outline:hover{border-color:var(--fg)}
input{border:1px solid var(--border);border-radius:var(--radius);padding:.7rem 1rem;font-size:1rem;
  color:var(--fg);background:var(--bg-card);transition:border-color var(--transition)}
input:focus{outline:none;border-color:var(--accent)}
.divider{width:3rem;height:1px;background:var(--border);margin:var(--space) 0}""",
    },
]


def select_references(description: str, scaffold_id: str, max_refs: int = 7) -> str:
    """Select the most relevant reference implementations for a build.

    Uses vector similarity (fastembed) when available for semantic matching,
    falls back to keyword scoring otherwise.

    Strategy:
    1. Always include references marked ``always_for`` this scaffold
    2. Vector-rank (or keyword-rank) remaining by relevance to description
    3. Pick top matches up to ``max_refs`` total

    Returns a formatted string ready for injection into the working document.
    """
    # Filter to references that apply to this scaffold
    candidates = [r for r in REFERENCES if scaffold_id in r["scaffolds"]]
    always = [r for r in candidates if scaffold_id in r.get("always_for", [])]
    rest = [r for r in candidates if r not in always]

    remaining_slots = max_refs - len(always)
    if remaining_slots <= 0:
        selected = always[:max_refs]
    else:
        ranked = _vector_rank(description, rest, remaining_slots)
        selected = always + ranked

    if not selected:
        return ""

    parts = []
    for ref in selected:
        parts.append(f"### {ref['label']}\n```\n{ref['code']}\n```")

    return "## Reference Implementations\nFollow these patterns for structure, security, and style:\n\n" + "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Vector-based semantic matching (with keyword fallback)
# ---------------------------------------------------------------------------

_embed_model = None
_ref_embeddings: dict[str, list[float]] = {}


def _get_embed_model():
    """Lazy-load the embedding model (fastembed)."""
    global _embed_model
    if _embed_model is None:
        try:
            from fastembed import TextEmbedding
            _embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        except ImportError:
            _embed_model = False  # Mark as unavailable
    return _embed_model if _embed_model is not False else None


def _embed_text(text: str) -> list[float] | None:
    """Embed a single text string. Returns None if embedding unavailable."""
    model = _get_embed_model()
    if not model:
        return None
    return list(list(model.embed([text]))[0])


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _get_ref_embedding(ref: dict) -> list[float] | None:
    """Get or compute the embedding for a reference (cached in memory)."""
    ref_id = ref["id"]
    if ref_id not in _ref_embeddings:
        # Embed the label + keywords as the reference's semantic signature
        text = ref["label"] + " " + " ".join(ref.get("keywords", []))
        emb = _embed_text(text)
        if emb:
            _ref_embeddings[ref_id] = emb
    return _ref_embeddings.get(ref_id)


def _vector_rank(description: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Rank candidates by vector similarity to description.

    Falls back to keyword scoring if embedding model unavailable.
    """
    query_emb = _embed_text(description)

    if query_emb:
        # Vector similarity ranking
        scored = []
        for ref in candidates:
            ref_emb = _get_ref_embedding(ref)
            if ref_emb:
                sim = _cosine_sim(query_emb, ref_emb)
                scored.append((sim, ref))
            else:
                scored.append((0.0, ref))
        scored.sort(key=lambda x: -x[0])
        return [ref for _, ref in scored[:top_k]]

    # Fallback: keyword scoring
    desc_lower = description.lower()
    scored = []
    for ref in candidates:
        score = sum(1 for kw in ref.get("keywords", []) if kw in desc_lower)
        if score > 0:
            scored.append((score, ref))
    scored.sort(key=lambda x: -x[0])
    result = [ref for _, ref in scored[:top_k]]

    # Fill remaining slots if needed
    if len(result) < min(top_k, 2):
        for ref in candidates:
            if ref not in result:
                result.append(ref)
                if len(result) >= min(top_k, 2):
                    break

    return result

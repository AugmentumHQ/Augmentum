/* ==========================================================================
   Agent-bridge shim — injected INTO a generated game's iframe document so the
   game_agent can play it autonomously.

   This is the browser half of the js13k/bridged surface (the Python half —
   BridgedAdapter + the WS route in proxy/game_agent_routes.py — already
   exists). It rides the exact bundle-composer injection path the save-bridge
   uses, so it is canvas-agnostic: it works for a 2D <canvas> game AND a
   WebGL/three.js game rendering a Blender-made GLB. "Speak both" for free.

   Responsibilities (all inside the sandboxed game document):
     1. Open a WebSocket to the session's bridge endpoint.
     2. Turn inbound {action, duration_ms} into synthetic KeyboardEvents
        using the game's declared semantic_to_key map.
     3. Sample the game <canvas> to PNG at a steady cadence and push frames.
     4. Intercept the game's own postMessage(screen/progress/won) and forward
        it as observation events the score is derived from.

   The config is baked at compose time as window.__AGENT_BRIDGE__.
   ========================================================================== */

/**
 * Build the <script> string injected into the iframe srcdoc when a game is
 * composed in agent-play mode.
 *
 * @param {object} cfg
 * @param {string} cfg.wsUrl        Absolute or origin-relative bridge WS URL.
 * @param {string} cfg.sessionId    game-agent session id.
 * @param {string} cfg.token        Per-session bearer token (query param).
 * @param {Object<string,string>} [cfg.semanticToKey]  Fallback control map,
 *   used only if the game does not declare AUGMENTUM_GAME.semantic_to_key.
 * @param {number} [cfg.frameHz]    Frame push cadence (default 2 Hz).
 * @returns {string} A complete <script>…</script> block.
 */
export function agentBridgeShim(cfg) {
  const config = {
    wsUrl: cfg.wsUrl || '',
    sessionId: cfg.sessionId || '',
    token: cfg.token || '',
    semanticToKey: cfg.semanticToKey || {},
    frameHz: cfg.frameHz || 2,
  };
  const literal = JSON.stringify(config);
  // The body is a template string with NO outer-scope interpolation beyond
  // the config literal — everything else runs in the iframe at play time.
  return `<script>/* augmentum agent-bridge */
(function(){
  var CFG = ${literal};
  window.__AGENT_BRIDGE__ = CFG;

  // ── Control map: prefer the game's own declaration, fall back to CFG. ──
  function controlMap(){
    try {
      var g = window.AUGMENTUM_GAME;
      if (g && g.semantic_to_key && typeof g.semantic_to_key === 'object') {
        return g.semantic_to_key;
      }
    } catch (e) {}
    return CFG.semanticToKey || {};
  }

  // ── Synthetic input: dispatch keydown, hold, keyup for a semantic. ──
  function keyProps(code){
    // Best-effort KeyboardEvent init from a KeyboardEvent.code. key/keyCode
    // are filled for the common arrows + space so games reading any of the
    // three properties respond.
    var map = {
      ArrowLeft:{key:'ArrowLeft',keyCode:37}, ArrowRight:{key:'ArrowRight',keyCode:39},
      ArrowUp:{key:'ArrowUp',keyCode:38}, ArrowDown:{key:'ArrowDown',keyCode:40},
      Space:{key:' ',keyCode:32}, Enter:{key:'Enter',keyCode:13},
    };
    var m = map[code] || {key:code, keyCode:0};
    return { code:code, key:m.key, keyCode:m.keyCode, which:m.keyCode, bubbles:true, cancelable:true };
  }
  function fire(type, code){
    var props = keyProps(code); props.type = type;
    var ev;
    try { ev = new KeyboardEvent(type, props); }
    catch(e){ ev = document.createEvent('Event'); ev.initEvent(type, true, true);
              ev.code=code; ev.key=props.key; ev.keyCode=props.keyCode; ev.which=props.which; }
    // Dispatch on document AND window — games bind to either.
    try { document.dispatchEvent(ev); } catch(e){}
    try { window.dispatchEvent(ev); } catch(e){}
    var c = document.querySelector('canvas');
    if (c) { try { c.dispatchEvent(ev); } catch(e){} }
  }
  function doAction(action, durationMs){
    var code = controlMap()[action];
    if (!code) return;
    fire('keydown', code);
    setTimeout(function(){ fire('keyup', code); }, Math.max(10, durationMs||120));
  }

  // ── WebSocket to the bridge. ──
  var ws = null, wsOpen = false, frameTimer = null;
  function connect(){
    var url = CFG.wsUrl;
    if (url && url.indexOf('://') === -1) {
      var proto = (location.protocol === 'https:') ? 'wss:' : 'ws:';
      url = proto + '//' + location.host + (url[0]==='/'?'':'/') + url;
    }
    if (CFG.token) url += (url.indexOf('?')===-1?'?':'&') + 'token=' + encodeURIComponent(CFG.token);
    try { ws = new WebSocket(url); } catch(e){ return; }
    ws.onopen = function(){ wsOpen = true; startFrames(); };
    ws.onclose = function(){ wsOpen = false; stopFrames(); };
    ws.onerror = function(){};
    ws.onmessage = function(ev){
      var msg; try { msg = JSON.parse(ev.data); } catch(e){ return; }
      if (msg && msg.action === 'request_frame') { pushFrame(); return; }
      if (msg && msg.action) { doAction(msg.action, msg.duration_ms); }
    };
  }
  function send(obj){ if (wsOpen && ws) { try { ws.send(JSON.stringify(obj)); } catch(e){} } }

  // ── Frames: sample the canvas to PNG and push. ──
  function pushFrame(){
    var c = document.querySelector('canvas');
    if (!c) return;
    var url;
    try { url = c.toDataURL('image/png'); } catch(e){ return; } // tainted canvas → skip
    var b64 = url.indexOf(',') >= 0 ? url.slice(url.indexOf(',')+1) : url;
    send({ kind:'frame', png_b64: b64 });
  }
  function startFrames(){
    stopFrames();
    var period = Math.max(200, Math.round(1000 / (CFG.frameHz||2)));
    frameTimer = setInterval(pushFrame, period);
    pushFrame();
  }
  function stopFrames(){ if (frameTimer){ clearInterval(frameTimer); frameTimer=null; } }

  // ── Observation events: intercept the game's postMessage to the host. ──
  var FORWARD = { screen:1, progress:1, won:1, score:1 };
  function forward(msg){
    if (msg && typeof msg === 'object' && FORWARD[msg.type]) {
      send({ kind:'event', data: msg });
    }
  }
  function wrap(target){
    if (!target || !target.postMessage) return;
    var orig = target.postMessage.bind(target);
    target.postMessage = function(message, targetOrigin, transfer){
      try { forward(message); } catch(e){}
      return orig(message, targetOrigin, transfer);
    };
  }
  try { wrap(window.parent); } catch(e){}
  try { wrap(window); } catch(e){}

  if (document.readyState === 'complete' || document.readyState === 'interactive') connect();
  else window.addEventListener('DOMContentLoaded', connect);
})();
</script>`;
}

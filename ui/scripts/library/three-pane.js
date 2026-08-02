/**
 * library/three-pane.js — reusable three-pane layout (sidebar / main / detail).
 *
 * Shape mirrors Linear / Things / Notion: fixed-width sidebar on the left,
 * flexible main pane in the middle, fixed-width detail pane on the right.
 * Each pane is just a host <div> — content rendering is the caller's job.
 *
 * On mobile the layout collapses to a single-pane stack with a drawer
 * for the sidebar and a bottom sheet for the detail pane. The substrate
 * exposes navigation methods (showMain / showDetail / showSidebar) so
 * the caller doesn't need to know which form factor it's running in.
 *
 *   const { sidebar, main, detail, showDetail } = createThreePane(host);
 *   sidebar.append(...);
 *   main.append(...);
 *   detail.append(...);
 *   // On mobile, surface detail when an item is selected:
 *   showDetail();
 */

export function createThreePane(host) {
  host.classList.add('lib-three-pane');

  const sidebar = document.createElement('aside');
  sidebar.className = 'lib-pane lib-pane-sidebar';
  sidebar.setAttribute('aria-label', 'Library navigation');

  const main = document.createElement('section');
  main.className = 'lib-pane lib-pane-main';
  main.setAttribute('aria-label', 'Library items');

  const detail = document.createElement('aside');
  detail.className = 'lib-pane lib-pane-detail';
  detail.setAttribute('aria-label', 'Selected item');

  host.replaceChildren(sidebar, main, detail);

  // ── Mobile sheet/drawer handlers ────────────────────────────────────

  function setVisiblePane(name) {
    host.dataset.visiblePane = name;  // 'sidebar' | 'main' | 'detail'
  }
  setVisiblePane('main');

  function showSidebar() { setVisiblePane('sidebar'); }
  function showMain() { setVisiblePane('main'); }
  function showDetail() { setVisiblePane('detail'); }

  function getVisiblePane() {
    return host.dataset.visiblePane || 'main';
  }

  return {
    host, sidebar, main, detail,
    showSidebar, showMain, showDetail, getVisiblePane,
  };
}

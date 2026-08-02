document.addEventListener("DOMContentLoaded", () => {
  if (typeof mermaid !== "undefined") {
    mermaid.initialize({ startOnLoad: false, theme: "dark", securityLevel: "strict" });
  }
});

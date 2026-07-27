(function () {
  var STORAGE_KEY = "framebid_site_unlock";
  var UNLOCK_VALUE = "lvlupdigi-ok";
  var PASSWORD = "LVLUPDIGI";

  function unlocked() {
    try {
      return localStorage.getItem(STORAGE_KEY) === UNLOCK_VALUE;
    } catch (e) {
      return false;
    }
  }

  function unlock() {
    try {
      localStorage.setItem(STORAGE_KEY, UNLOCK_VALUE);
    } catch (e) {}
  }

  function isAccessPage() {
    var path = (location.pathname || "").replace(/\/+$/, "") || "/";
    return path === "/access" || /\/access\.html$/i.test(path);
  }

  function nextTarget() {
    var params = new URLSearchParams(location.search);
    var next = params.get("next") || "/signin";
    if (!next.startsWith("/") || next.startsWith("//") || next.indexOf("/access") === 0) {
      return "/signin";
    }
    // Don't send people back to the public landing after unlocking
    if (next === "/" || next === "/index.html" || next === "/landing.html") {
      return "/signin";
    }
    return next;
  }

  window.FrameBidGate = {
    unlocked: unlocked,
    unlock: unlock,
    checkPassword: function (value) {
      return String(value || "") === PASSWORD;
    },
    nextTarget: nextTarget,
  };

  if (unlocked() || isAccessPage()) return;

  var next = location.pathname + location.search + location.hash;
  if (!next || next === "/access") next = "/signin";
  location.replace("/access?next=" + encodeURIComponent(next));
})();

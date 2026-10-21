// Light/dark mode. Bootstrap 5.3 themes everything from <html data-bs-theme>;
// style.css restyles the Plotly figures under the same attribute.
// Default: the system preference, followed live. A click on #theme-toggle
// stores an explicit choice, which then wins over the system setting.
(function () {
    var KEY = 'robopid-theme';
    var media = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

    function stored() {
        try { return localStorage.getItem(KEY); } catch (e) { return null; }
    }
    function system() { return media && media.matches ? 'dark' : 'light'; }
    function apply(theme) { document.documentElement.setAttribute('data-bs-theme', theme); }

    apply(stored() || system());          // before Dash renders: no light flash

    if (media) {
        var follow = function () { if (!stored()) apply(system()); };
        if (media.addEventListener) media.addEventListener('change', follow);
        else if (media.addListener) media.addListener(follow);
    }

    // The button is rendered by Dash after this script runs, so listen on document.
    document.addEventListener('click', function (ev) {
        var btn = ev.target.closest && ev.target.closest('#theme-toggle');
        if (!btn) return;
        var next = document.documentElement.getAttribute('data-bs-theme') === 'dark' ? 'light' : 'dark';
        try { localStorage.setItem(KEY, next); } catch (e) { /* private mode: still switch */ }
        apply(next);
    });
})();

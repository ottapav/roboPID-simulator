// Makes each graph's "Download plot as a png" modebar button prompt for a
// save location (via the File System Access API) instead of silently
// dropping the file into the browser's default downloads folder.
// Browsers without showSaveFilePicker (Firefox, Safari) fall back to
// Plotly's normal, dialog-less download.
(function () {
    function findDownloadButton(el) {
        return el.closest('[data-title="Download plot as a png"]');
    }

    async function saveWithPicker(gd) {
        let handle;
        try {
            handle = await window.showSaveFilePicker({
                suggestedName: 'plot.png',
                types: [{description: 'PNG image', accept: {'image/png': ['.png']}}],
            });
        } catch (err) {
            return; // user cancelled the picker
        }
        const width = gd._fullLayout ? gd._fullLayout.width : undefined;
        const height = gd._fullLayout ? gd._fullLayout.height : undefined;
        const dataUrl = await window.Plotly.toImage(gd, {format: 'png', width, height});
        const blob = await (await fetch(dataUrl)).blob();
        const writable = await handle.createWritable();
        await writable.write(blob);
        await writable.close();
    }

    document.addEventListener('click', function (event) {
        if (!window.showSaveFilePicker) return; // unsupported browser: keep default behavior
        const btn = findDownloadButton(event.target);
        if (!btn) return;
        const gd = btn.closest('.js-plotly-plot');
        if (!gd) return;
        event.stopImmediatePropagation();
        event.preventDefault();
        saveWithPicker(gd);
    }, true);
})();

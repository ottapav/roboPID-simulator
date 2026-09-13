// Makes each graph's "Download plot as a png" modebar button prompt for a
// save location (via the File System Access API) instead of silently
// dropping the file into the browser's default downloads folder.
// The picker offers SVG (vector) or PNG rendered at PNG_SCALE× the on-screen
// size; the format follows the extension of the chosen file name.
// Browsers without showSaveFilePicker (Firefox, Safari) fall back to
// Plotly's normal, dialog-less download (scaled via toImageButtonOptions).
(function () {
    const PNG_SCALE = 4;

    function findDownloadButton(el) {
        return el.closest('[data-title="Download plot as a png"]');
    }

    async function saveWithPicker(gd) {
        let handle;
        try {
            handle = await window.showSaveFilePicker({
                suggestedName: 'plot.svg',
                types: [
                    {description: 'SVG image', accept: {'image/svg+xml': ['.svg']}},
                    {description: 'PNG image', accept: {'image/png': ['.png']}},
                ],
            });
        } catch (err) {
            return; // user cancelled the picker
        }
        const format = handle.name.toLowerCase().endsWith('.svg') ? 'svg' : 'png';
        const width = gd._fullLayout ? gd._fullLayout.width : undefined;
        const height = gd._fullLayout ? gd._fullLayout.height : undefined;
        const scale = format === 'png' ? PNG_SCALE : 1;
        const dataUrl = await window.Plotly.toImage(gd, {format, width, height, scale});
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

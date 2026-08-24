window.dccFunctions = window.dccFunctions || {};

// The GUI shows every number with two digits after the decimal point. This is
// core.params.fmt2 in JavaScript, exponential branch included: the gain box
// reaches down to 1e-4, and a tooltip reading "0.00" while the slider sits at a
// live gain would misreport it rather than merely round it.
window.dccFunctions.fmt2 = function (value) {
    if (value !== 0 && Math.abs(value) < 0.005) {
        return value.toExponential(2);
    }
    return value.toFixed(2);
};

// Sliders hold log10(gain); the tooltip shows the absolute gain it stands for.
window.dccFunctions.logGain = function (value) {
    return window.dccFunctions.fmt2(Math.pow(10, value));
};

window.dccFunctions = window.dccFunctions || {};
window.dccFunctions.round3dp = function (value) {
    return Math.round(value * 1000) / 1000;
};
window.dccFunctions.logGain = function (value) {
    return Math.round(Math.pow(10, value) * 1000) / 1000;
};

/**
 * antelligent.js: the antelligent mark, eighteen gold dots tracing an infinity,
 * mounted once as an SVG <symbol> that the page draws with
 *
 *   <svg viewBox="0 0 159.6 95.5"><use href="#ant-mark"/></svg>
 *
 * Inline rather than an .svg file so the page needs nothing from the network.
 * The dots are fixed gold (#DFAD43) and do not follow the theme.
 */
(function () {
  'use strict';

  /* [cx, cy, r] of each dot, in the order the site's file lists them. */
  var DOTS = [
    [146.3, 59.7, 4.5], [133.9, 78.3, 6.7], [111.9, 80.2, 8.7], [93.8, 66.6, 10.1],
    [79.8, 48.4, 10.9], [65.8, 30.4, 10.9], [47.7, 16.7, 10.1], [25.8, 18.5, 8.7],
    [13.3, 37.1, 6.7], [13.3, 59.7, 4.5], [25.7, 78.3, 6.7], [47.7, 80.2, 8.7],
    [65.8, 66.6, 10.1], [79.8, 48.4, 10.9], [93.8, 30.4, 10.9], [111.9, 16.7, 10.1],
    [133.8, 18.5, 8.7], [146.3, 37.1, 6.7]
  ];

  function mount() {
    if (document.getElementById('ant-mark')) return;
    var circles = DOTS.map(function (d) {
      return '<circle cx="' + d[0] + '" cy="' + d[1] + '" r="' + d[2] + '"/>';
    }).join('');
    var t = document.createElement('template');
    t.innerHTML = '<svg width="0" height="0" style="position:absolute" aria-hidden="true">' +
      '<symbol id="ant-mark" viewBox="0 0 159.6 95.5" fill="#DFAD43">' + circles + '</symbol></svg>';
    document.body.insertBefore(t.content.firstElementChild, document.body.firstChild);
  }

  /* Normally this runs from the end of <body>, where the body already exists.
     The fallback is for an artifact that loads it from <head>. */
  if (document.body) mount();
  else document.addEventListener('DOMContentLoaded', mount);
})();

// Shared draggable detail windows. Graph and species windows use stable keys:
// reopening an existing item focuses it instead of stacking duplicates.

import { makeResizable, clampToViewport } from './resizable.js';

let nextZ = 100;
const open = new Map();

// Margin kept between a window and the edge of the viewport, and the share of
// the viewport a window may claim on open. A window that opens larger than the
// screen is worse than one that opens small: the small one can be dragged
// bigger, the large one has already covered the pond.
const EDGE = 16;
const MAX_SHARE = 0.9;

export function openFloating({ key, title, className = '', render, size = {} }) {
    const existing = open.get(key);
    if (existing) {
        focus(existing);
        render(existing.querySelector('.float-body'));
        return existing;
    }

    const win = document.createElement('section');
    win.className = `float-window ${className}`.trim();
    win.innerHTML = `
        <header class="float-head">
            <span class="float-title"></span>
            <button class="float-close" aria-label="close">×</button>
        </header>
        <div class="float-body"></div>`;
    win.querySelector('.float-title').textContent = title;
    document.body.appendChild(win);

    // Explicit size rather than the content's own: `width: min-content` let the
    // phylogeny canvas, which is laid out at a fixed 780px and grows with the
    // roster, open wider and taller than a small window — `max-width` only
    // turned that into scrollbars. It is also what makes the resize grip
    // predictable; a min-content box jumps on the first drag.
    resize_to_fit(win, size);
    focus(win);

    win.querySelector('.float-close').addEventListener('click', () => {
        open.delete(key);
        win.remove();
    });
    win.addEventListener('mousedown', () => focus(win));
    makeDraggable(win, win.querySelector('.float-head'));
    // Re-render on resize. Canvas content is drawn at a fixed pixel size, so a
    // window that grows would otherwise just add empty space around the chart.
    makeResizable(win, {
        corner: 'se',
        minW: size.minW ?? 300,
        minH: size.minH ?? 140,
        onResize: () => {
            // A window the user has sized owns its height; the stylesheet's
            // max-height would otherwise clip a deliberately tall one.
            render(win.querySelector('.float-body'));
        },
    });
    render(win.querySelector('.float-body'));

    // Position last: a window whose height comes from its content has no height
    // until it has been rendered, and clamping against zero put every window at
    // the cascade offset regardless of how tall it turned out to be.
    const count = open.size;
    win.style.left = `${Math.max(EDGE, Math.min(window.innerWidth - win.offsetWidth - EDGE, 90 + count * 28))}px`;
    win.style.top = `${Math.max(EDGE, Math.min(window.innerHeight - win.offsetHeight - EDGE, 70 + count * 24))}px`;
    clamp(win);

    open.set(key, win);
    return win;
}

/** The z-index of the frontmost window. Anything that has to sit over the
 *  windows — a tooltip hanging off one — reads this rather than picking a
 *  number, since `focus()` raises the counter without bound and a fixed number
 *  loses as soon as a window has been clicked a few times. */
export function topZ() {
    return nextZ;
}

export function isFloatingOpen(key) {
    return open.has(key);
}

/** Close one window by key. Returns whether there was one to close, so a caller
 *  can use it as the "off" half of a toggle without asking twice. */
export function closeFloating(key) {
    const win = open.get(key);
    if (!win) return false;
    open.delete(key);
    win.remove();
    return true;
}

export function updateFloating(key, render) {
    const win = open.get(key);
    if (win) render(win.querySelector('.float-body'));
}

export function closeFloatingPrefix(prefix) {
    for (const [key, win] of [...open.entries()]) {
        if (!key.startsWith(prefix)) continue;
        open.delete(key);
        win.remove();
    }
}

function focus(win) {
    win.style.zIndex = String(++nextZ);
}

/** Size a window to its preferred size, capped by what the viewport can hold.
 *  `size` is `{w, h, minW}` in CSS px; anything missing gets a default.
 *
 *  Width is always explicit — it is what the content lays out against, and
 *  leaving it to `min-content` is how a fixed-width canvas came to open wider
 *  than the screen. Height is left to the content unless a caller asks for one:
 *  a short card should be short, and the `max-height` in the stylesheet already
 *  stops a tall one from running off the bottom. */
function resize_to_fit(win, size = {}) {
    const minW = size.minW ?? 300;
    const availW = window.innerWidth - EDGE * 2;
    const w = Math.max(Math.min(minW, availW), Math.min(size.w ?? 420, availW));
    win.style.width = `${Math.round(w)}px`;
    if (size.h) {
        const availH = Math.min(window.innerHeight - EDGE * 2, window.innerHeight * MAX_SHARE);
        win.style.height = `${Math.round(Math.min(size.h, availH))}px`;
    }
}

function clamp(win) {
    const margin = 8;
    const maxX = Math.max(margin, window.innerWidth - win.offsetWidth - margin);
    const maxY = Math.max(margin, window.innerHeight - win.offsetHeight - margin);
    const x = Math.max(margin, Math.min(parseFloat(win.style.left) || margin, maxX));
    const y = Math.max(margin, Math.min(parseFloat(win.style.top) || margin, maxY));
    win.style.left = `${x}px`;
    win.style.top = `${y}px`;
}

function makeDraggable(win, handle) {
    let drag = null;
    handle.addEventListener('mousedown', e => {
        if (e.button !== 0 || e.target.closest('button')) return;
        const rect = win.getBoundingClientRect();
        drag = { dx: e.clientX - rect.left, dy: e.clientY - rect.top };
        focus(win);
        e.preventDefault();
    });
    window.addEventListener('mousemove', e => {
        if (!drag) return;
        win.style.left = `${e.clientX - drag.dx}px`;
        win.style.top = `${e.clientY - drag.dy}px`;
        clamp(win);
    });
    window.addEventListener('mouseup', () => { drag = null; });
    // Size first (makeResizable's own listener may also have shrunk it), then
    // position: where it can sit depends on how big it is.
    window.addEventListener('resize', () => { clampToViewport(win, 240, 140); clamp(win); });
}

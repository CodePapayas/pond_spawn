// Drag-to-resize for panels and floating windows.
//
// CSS `resize: both` was the obvious answer and it does not work here. The
// browser paints its grip in the element's bottom-right corner, but any child
// that reaches that corner takes the mouse first — which is every panel in this
// UI, since they are all a scrolling body filling the box. The grip was there;
// the pointer never got to it.
//
// So the grip is ours: a real element, on top, with its own drag. That also
// buys the two things the CSS property cannot do — a bottom-left grip for the
// panels anchored to the right edge (a right-anchored panel grows the wrong way
// from a bottom-right corner), and a resize callback, which the canvas panels
// need because their contents are drawn at a pixel size rather than laid out.

const MARGIN = 8;   // keep this much of the viewport clear of a resized edge

/**
 * Make `el` resizable by dragging a corner grip.
 *
 * @param {HTMLElement} el
 * @param {object}   opts
 * @param {'se'|'sw'} opts.corner   which corner carries the grip. Use 'sw' for
 *                                  anything anchored to the right edge.
 * @param {number}   opts.minW
 * @param {number}   opts.minH
 * @param {() => void} [opts.onResize]  called while dragging, rAF-coalesced.
 */
export function makeResizable(el, { corner = 'se', minW = 220, minH = 120, onResize } = {}) {
    const grip = document.createElement('div');
    // Where the grip can sit depends on who does the scrolling. A floating
    // window scrolls its inner body, so an absolutely positioned grip stays
    // pinned to the frame. A panel scrolls itself, and an absolute child of a
    // scroll container scrolls away with the content — so it sticks instead,
    // which pins it to the bottom of the visible area for free.
    const scrolls = /auto|scroll/.test(getComputedStyle(el).overflowY);
    grip.className = `resize-grip grip-${corner} ${scrolls ? 'grip-sticky' : 'grip-abs'}`;
    grip.title = 'drag to resize';
    el.appendChild(grip);

    let drag = null;
    let queued = false;

    const notify = () => {
        if (!onResize || queued) return;
        queued = true;
        requestAnimationFrame(() => { queued = false; onResize(); });
    };

    grip.addEventListener('mousedown', e => {
        if (e.button !== 0) return;
        // A centred panel (#graphs is `left:50%` + `translateX(-50%)`) recentres
        // itself as it grows, so the corner runs away from the cursor at half
        // the drag speed. Pin it where it is on the first drag and resize from
        // there; it stays where the user put it afterwards, which is the
        // behaviour a dragged panel should have anyway.
        if (getComputedStyle(el).transform !== 'none') {
            const r = el.getBoundingClientRect();
            el.style.left = `${r.left}px`;
            el.style.top = `${r.top}px`;
            el.style.bottom = 'auto';
            el.style.transform = 'none';
        }
        const rect = el.getBoundingClientRect();
        drag = { x: e.clientX, y: e.clientY, w: rect.width, h: rect.height,
                 left: rect.left, top: rect.top };
        // Otherwise the window's own drag handler starts a move at the same time.
        e.preventDefault();
        e.stopPropagation();
        document.body.style.userSelect = 'none';
    });

    window.addEventListener('mousemove', e => {
        if (!drag) return;
        const dx = e.clientX - drag.x;
        const dy = e.clientY - drag.y;

        // Ceilings are what the element can grow to without leaving the
        // viewport, given where its fixed edge is.
        const maxW = corner === 'se'
            ? window.innerWidth - drag.left - MARGIN
            : drag.left + drag.w - MARGIN;
        const maxH = window.innerHeight - drag.top - MARGIN;

        const w = Math.max(minW, Math.min(maxW, drag.w + (corner === 'se' ? dx : -dx)));
        const h = Math.max(minH, Math.min(maxH, drag.h + dy));
        el.style.width = `${Math.round(w)}px`;
        el.style.height = `${Math.round(h)}px`;
        // A resized panel owns its height, so the stylesheet's max-height must
        // stop overriding it — otherwise dragging taller than the cap silently
        // does nothing.
        el.style.maxHeight = 'none';
        // A left-growing grip keeps the right edge still. Windows are positioned
        // with `left`, so that edge has to be moved by hand.
        if (corner === 'sw' && el.style.left) {
            el.style.left = `${Math.round(drag.left + drag.w - w)}px`;
        }
        notify();
    });

    window.addEventListener('mouseup', () => {
        if (!drag) return;
        drag = null;
        document.body.style.userSelect = '';
        if (onResize) onResize();
    });

    // A panel dragged big and then a browser window dragged small is the case
    // that puts a menu over the whole pond, which is the thing being fixed.
    window.addEventListener('resize', () => {
        if (clampToViewport(el, minW, minH) && onResize) onResize();
    });

    return grip;
}

/** Shrink an element that no longer fits the viewport. Returns whether it
 *  changed anything, so callers can skip a redraw. */
export function clampToViewport(el, minW = 220, minH = 120) {
    const rect = el.getBoundingClientRect();
    let changed = false;
    const maxW = window.innerWidth - MARGIN * 2;
    const maxH = window.innerHeight - MARGIN * 2;
    if (rect.width > maxW) {
        el.style.width = `${Math.max(minW, maxW)}px`;
        changed = true;
    }
    if (rect.height > maxH) {
        el.style.height = `${Math.max(minH, maxH)}px`;
        changed = true;
    }
    return changed;
}

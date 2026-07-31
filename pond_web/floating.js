// Shared draggable detail windows. Graph and species windows use stable keys:
// reopening an existing item focuses it instead of stacking duplicates.

let nextZ = 100;
const open = new Map();

export function openFloating({ key, title, className = '', render }) {
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

    const count = open.size;
    win.style.left = `${Math.min(window.innerWidth - 360, 90 + count * 28)}px`;
    win.style.top = `${Math.min(window.innerHeight - 260, 70 + count * 24)}px`;
    focus(win);
    clamp(win);

    win.querySelector('.float-close').addEventListener('click', () => {
        open.delete(key);
        win.remove();
    });
    win.addEventListener('mousedown', () => focus(win));
    makeDraggable(win, win.querySelector('.float-head'));
    // Re-render on resize. Canvas content is drawn at a fixed pixel size, so a
    // window that grows would otherwise just add empty space around the chart.
    if (typeof ResizeObserver === 'function') {
        let last = 0;
        const ro = new ResizeObserver(() => {
            // Coalesce: a drag fires this continuously, and each render rebuilds
            // a canvas and repaints a few hundred samples.
            const now = performance.now();
            if (now - last < 60) return;
            last = now;
            render(win.querySelector('.float-body'));
        });
        ro.observe(win);
    }
    render(win.querySelector('.float-body'));
    open.set(key, win);
    return win;
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
    window.addEventListener('resize', () => clamp(win));
}

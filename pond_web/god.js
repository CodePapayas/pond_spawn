// God mode: player powers that overrule the simulation's rules.
//
// This module owns the panel, the armed-tool state, and the timed effects
// (spreading salt, the sweep wipe). It does not draw — the renderer asks it for
// the active effects each frame and paints them, so all canvas work stays in one
// place. Kills themselves go through pond_core, which tallies them as `Smitten`
// rather than laundering them through a natural cause.

export const TOOLS = {
    COMET: 'comet',
    SALT: 'salt',
};

// Comet: instant, one blast radius in world units.
const COMET_RADIUS = 2.2;
const COMET_SEC = 0.9;          // impact flash duration

// Salt: kills in a widening ring rather than all at once, so a pour reads as
// spreading through the water instead of as a second, slower comet.
const SALT_SEC = 5.0;
const SALT_MAX_RADIUS = 5.5;
const SALT_TICK_MS = 90;        // how often the kill ring advances

// Sweep: a column crossing the pond left to right, killing as it passes.
const SWEEP_SEC = 1.8;

/**
 * @param {HTMLElement} root  the #god panel
 * @param {object} api  { smiteRadius, smiteBand, smiteAll, setImmortal,
 *                        gridSize, onChange }
 */
export function initGod(root, api) {
    let enabled = false;
    let armed = null;           // TOOLS.* while a click-to-place tool is selected
    let immortal = false;

    // Active visual effects, read by the renderer each frame.
    const comets = [];          // {x, y, t0}
    const salts = [];           // {x, y, t0, lastTick, killedTo}
    let sweep = null;           // {t0, killedTo}

    root.innerHTML =
        `<h2><label class="god-toggle">` +
        `<input type="checkbox" id="god-enable"> god mode</label></h2>` +
        `<div id="god-body" class="god-body">` +
        `<div class="god-row"><button id="god-comet" class="god-tool">comet</button>` +
        `<span class="god-hint">click the pond — kills a radius</span></div>` +
        `<div class="god-row"><button id="god-salt" class="god-tool">salt</button>` +
        `<span class="god-hint">click the pond — spreads outward, killing</span></div>` +
        `<div class="god-row"><button id="god-sweep" class="god-tool">sweep clean</button>` +
        `<span class="god-hint">wipes the pond empty</span></div>` +
        `<div class="god-row"><button id="god-octagon" class="god-tool">octagon</button>` +
        `<span class="god-hint">unkillable hunter — eats until 1/5 remain, then leaves</span></div>` +
        `<div class="god-row"><button id="god-rectangle" class="god-tool">R E C T A N G L E</button>` +
        `<span class="god-hint">rotating sweep — kills everything its edges touch, ` +
        `however armoured</span></div>` +
        `<div class="god-row"><button id="god-dismiss" class="god-tool">dismiss hunters</button>` +
        `<span class="god-hint">calls off every summoned hunt — they swim away</span></div>` +
        `<div class="god-row"><label class="god-toggle">` +
        `<input type="checkbox" id="god-immortal"> immortality</label></div>` +
        `<div class="god-hint">nothing dies of age, hunger or combat &middot; ` +
        `summoned hunters still bite</div>` +
        `<div class="god-row"><label class="god-toggle">` +
        `<input type="checkbox" id="god-predators" checked> ambient predators</label></div>` +
        `<div class="god-hint">one resident hunter, always in the water &middot; ` +
        `off leaves the pond to its own devices</div>` +
        `<div id="god-warn" class="god-warn"></div>` +
        `</div>`;

    const el = id => root.querySelector('#' + id);
    const enable_box = el('god-enable');
    const body = el('god-body');
    const immortal_box = el('god-immortal');
    const predators_box = el('god-predators');
    const dismiss_btn = el('god-dismiss');
    const warn = el('god-warn');
    const tool_buttons = {
        [TOOLS.COMET]: el('god-comet'),
        [TOOLS.SALT]: el('god-salt'),
    };

    function set_enabled(on) {
        enabled = on;
        body.style.display = on ? 'block' : 'none';
        root.classList.toggle('active', on);
        if (!on) {
            arm(null);
            if (immortal) set_immortal(false);
        }
        // Deliberately does *not* dismiss the hunters already in the water:
        // closing the panel to watch what a rectangle does should not delete it.
        refresh_dismiss();
        api.onChange?.();
    }

    /** Label the off switch with what it would act on, and disable it when there
     *  is no summoned hunt to call off. */
    function refresh_dismiss() {
        const n = api.summonedHunterCount();
        dismiss_btn.textContent = n > 0 ? `dismiss hunters (${n})` : 'dismiss hunters';
        dismiss_btn.disabled = n === 0;
    }

    function arm(tool) {
        armed = armed === tool ? null : tool;
        for (const [name, btn] of Object.entries(tool_buttons)) {
            btn.classList.toggle('armed', armed === name);
        }
        api.onChange?.();
    }

    function set_immortal(on) {
        immortal = on;
        immortal_box.checked = on;
        api.setImmortal(on);
        warn.innerHTML = on
            ? `<strong>warning:</strong> with nothing dying, the population only grows. ` +
              `High agent counts may cause severe performance issues or melt your ` +
              `computer. Use at your own risk.`
            : '';
    }

    enable_box.addEventListener('change', () => set_enabled(enable_box.checked));
    tool_buttons[TOOLS.COMET].addEventListener('click', () => arm(TOOLS.COMET));
    tool_buttons[TOOLS.SALT].addEventListener('click', () => arm(TOOLS.SALT));
    el('god-sweep').addEventListener('click', () => start_sweep());
    // The triangles are the pond's own answer to overpopulation and arrive on
    // their own. These two never do — they exist only here.
    el('god-octagon').addEventListener('click', () => {
        if (enabled) api.summonOctagon();
        refresh_dismiss();
    });
    el('god-rectangle').addEventListener('click', () => {
        if (enabled) api.summonRectangle();
        refresh_dismiss();
    });
    // The off switch for the two shapes above. The HUD's `predators` toggle is
    // the ecology's, and deliberately spares player summons, so without this an
    // octagon or a rectangle could only be waited out.
    el('god-dismiss').addEventListener('click', () => {
        if (enabled) api.dismissHunters();
        refresh_dismiss();
    });
    immortal_box.addEventListener('change', () => set_immortal(immortal_box.checked));
    // Registered here with the other listeners. It was inside `effects()`, which
    // the renderer calls every frame: a new listener was added per frame, and
    // `effects()` returned this object instead of the frame's effects.
    predators_box.addEventListener('change', () => {
        api.setAutomaticPredators(predators_box.checked);
    });

    set_enabled(false);

    // ── Tool use ──────────────────────────────────────────────────────────────

    /** Apply the armed tool at a world position. Returns true if it consumed
     *  the click, so the renderer knows not to also select or stir. */
    function use_at(wx, wy, now_sec) {
        if (!enabled || !armed) return false;
        if (armed === TOOLS.COMET) {
            api.smiteRadius(wx, wy, COMET_RADIUS);
            comets.push({ x: wx, y: wy, t0: now_sec });
        } else if (armed === TOOLS.SALT) {
            salts.push({ x: wx, y: wy, t0: now_sec, killedTo: 0 });
        }
        return true;
    }

    function start_sweep() {
        if (!enabled || sweep) return;
        sweep = { t0: null, killedTo: 0 };   // t0 set on the first frame it's drawn
        api.onChange?.();
    }

    /** Advance timed effects and apply their kills. Called once per frame by the
     *  renderer, before drawing. Kills are driven by wall-clock time, not sim
     *  steps, so a spreading salt patch keeps spreading while the sim is paused —
     *  god does not need the pond's permission. */
    function update(now_sec) {
        const grid = api.gridSize();

        // Hunters leave on their own when their cull is done, so the off switch
        // has to track the pond rather than only the player's clicks.
        if (enabled) refresh_dismiss();

        for (let i = comets.length - 1; i >= 0; i--) {
            if (now_sec - comets[i].t0 > COMET_SEC) comets.splice(i, 1);
        }

        for (let i = salts.length - 1; i >= 0; i--) {
            const s = salts[i];
            const age = now_sec - s.t0;
            if (age > SALT_SEC) { salts.splice(i, 1); continue; }
            const r = SALT_MAX_RADIUS * ease_out(age / SALT_SEC);
            // Kill on a coarse tick rather than every frame: the ring only has
            // to keep up with the visual, and each call is a full agent scan.
            if (r - s.killedTo > SALT_MAX_RADIUS * (SALT_TICK_MS / 1000) / SALT_SEC) {
                api.smiteRadius(s.x, s.y, r);
                s.killedTo = r;
            }
        }

        if (sweep) {
            if (sweep.t0 === null) sweep.t0 = now_sec;
            const age = now_sec - sweep.t0;
            const x = Math.min(grid, (age / SWEEP_SEC) * grid);
            if (x > sweep.killedTo) {
                api.smiteBand(sweep.killedTo, x);
                sweep.killedTo = x;
            }
            if (age > SWEEP_SEC) {
                api.smiteAll();   // catch anything spawned behind the wipe
                sweep = null;
                api.onChange?.();
            }
        }
    }

    /** Effects the renderer should draw this frame, in world coordinates. */
    function effects(now_sec) {
        return {
            comets: comets.map(c => ({
                x: c.x, y: c.y,
                t: (now_sec - c.t0) / COMET_SEC,
                radius: COMET_RADIUS,
            })),
            salts: salts.map(s => ({
                x: s.x, y: s.y,
                t: (now_sec - s.t0) / SALT_SEC,
                radius: SALT_MAX_RADIUS * ease_out((now_sec - s.t0) / SALT_SEC),
            })),
            sweep: sweep && sweep.t0 !== null
                ? { t: (now_sec - sweep.t0) / SWEEP_SEC }
                : null,
        };
    }

    return {
        update,
        effects,
        useAt: use_at,
        isEnabled: () => enabled,
        armedTool: () => armed,
        isImmortal: () => immortal,
        show() { root.style.display = 'block'; },
        hide() { root.style.display = 'none'; },
    };
}

function ease_out(t) {
    const c = Math.min(1, Math.max(0, t));
    return 1 - (1 - c) * (1 - c);
}

// Opening card: what this is, how to touch it, and the two ways out.
//
// The page used to open on the setup panel with the pond frozen behind it, so
// the first thing anyone saw was a form. A pond that is already running is a
// better argument for itself than any description of one, so the run starts on
// load and this card sits over the top of it — dismissable, and never shown
// again in the session.
//
// It owns nothing about the run. `continue` just hides it; `new run` hands off
// to the setup panel, which is where run parameters have always lived.

// Enough of the controls key to get someone moving, not all of it — the full
// list is bottom-left the moment this closes.
const KEYS = [
    ['click', 'inspect an agent — traits, brain, lineage'],
    ['drag', 'stir the water'],
    ['wheel / f', 'zoom · fit the pond'],
    ['space', 'pause'],
    ['+ / -', 'speed'],
    ['g · p · b', 'graphs · phylogeny · archetypes'],
    ['l · k', 'legend · controls key'],
    ['n', 'new run'],
    ['c', 'clear the interface'],
];

/**
 * @param {HTMLElement} root  the #splash panel
 * @param {{onContinue: () => void, onNewRun: () => void}} api
 */
export function initSplash(root, api) {
    root.innerHTML =
        `<h2>pond spawn</h2>` +
        `<div class="splash-blurb">Welcome to Pond Spawn! Please take a moment ` +
        `to familiarize yourself with the controls below. A simulation has ` +
        `already been started at a decently advanced point in its lifecycle. ` +
        `This is to demonstrate to you what a mature pond with successful ` +
        `speciation may look like. You can either keep watching this pond or ` +
        `you can start your own with the NEW RUN button below. Thanks, and ` +
        `have fun!</div>` +
        `<div class="splash-keys">` +
        KEYS.map(([k, what]) =>
            `<div class="splash-key"><b>${k}</b><span>${what}</span></div>`).join('') +
        `</div>` +
        `<div class="splash-status" id="splash-status"></div>` +
        `<div class="setup-actions">` +
        `<button id="splash-new">new run</button>` +
        `<button id="splash-go" class="primary">watch this pond</button>` +
        `</div>`;

    const status = root.querySelector('#splash-status');
    root.querySelector('#splash-go').addEventListener('click', () => {
        hide();
        api.onContinue();
    });
    root.querySelector('#splash-new').addEventListener('click', () => {
        hide();
        api.onNewRun();
    });

    function hide() { root.style.display = 'none'; }

    // Held back until the opening curtain lifts — both are centred, and a
    // welcome card sitting on top of "digging a pond for u ;)" is two things
    // competing for the same middle of the screen.
    root.style.display = 'none';

    return {
        isOpen() { return root.style.display === 'block'; },
        show() { root.style.display = 'block'; },
        hide,
        /** Progress line while the opening pond is being wound forward. Cleared
         *  by passing null — the card stays up, it just stops narrating. */
        setStatus(text) {
            status.textContent = text ?? '';
            status.style.display = text ? 'block' : 'none';
        },
    };
}

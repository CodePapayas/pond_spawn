//! Binomial names for promoted species.
//!
//! Deterministic from `(species_id, world_seed)` and nothing else — a replay of
//! the same seed produces the same names. The RNG is a private `ChaCha8Rng`, never
//! the world's stream: drawing from that would shift every subsequent simulation
//! draw and break determinism against existing traces.
//!
//! ## Genus — always feminine, and inherited
//!
//! Genus names are invented syllables with a feminine ending (*Vorixa*,
//! *Thalura*, *Mekasia*). One gender throughout means the genus reads as a
//! consistent family rather than as Latin-flavoured noise.
//!
//! A species promoting near an existing one — live *or* extinct — inherits that
//! genus and takes a new epithet. Deriving the genus from `species_id` alone
//! would give two species that split from a common ancestor unrelated genus
//! names, making the naming actively lie about descent. Inheritance means a
//! re-radiation after a bottleneck reads as one genus fanning out, which is what
//! actually happened.
//!
//! ## Epithet — Latin when the lineage specialized, nonsense when it did not
//!
//! The epithet is chosen from the species' **signed deviation from the
//! population centroid**, not from the largest trait value. Argmax over the
//! centroid just returns whichever trait sits high in the population as a whole,
//! so in an aggressive pond every species ends up *ferox* and the epithet says
//! nothing. Deviation names what makes this lineage *different*, and the sign
//! matters: a metabolism far below the pond is as much a specialization as one
//! far above, so each trait carries a high word and a low word.
//!
//! When the strongest deviation clears `STRONG_DEVIATION` the epithet is a real
//! Latin adjective naming that trait and direction. When nothing stands out it
//! is nonsense. So the name itself reports whether the lineage specialized at
//! all — *Suralia loricata* is armoured, *Suralia kyrnus* is simply Suralia.
//!
//! Epithet endings deliberately mix masculine and feminine (*-us* / *-a*) rather
//! than agreeing with the feminine genus as real Latin grammar would require.
//! The variety is worth more here than the agreement.

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::species::SIG_LEN;

/// Minimum absolute deviation from the population centroid, in normalized trait
/// space, for a species to be considered specialized enough to earn a Latin
/// epithet. Below this the epithet is nonsense.
pub const STRONG_DEVIATION: f64 = 0.12;

/// Genus opening syllables. Pure invention — the genus carries lineage identity,
/// not ecology, so it should not look like it means anything.
const GENUS_HEAD: [&str; 12] = [
    "vor", "thal", "mek", "sur", "ix", "kel",
    "dra", "phos", "nyr", "zan", "oth", "lum",
];
/// Optional middle syllable, giving three-syllable genera alongside two.
const GENUS_MID: [&str; 6] = ["ix", "al", "or", "en", "yr", "as"];
/// Feminine endings. Every genus takes one, without exception.
const GENUS_TAIL: [&str; 5] = ["a", "ia", "ura", "ana", "essa"];

/// Nonsense epithet stems, for species with no standout trait.
const NONSENSE_STEM: [&str; 12] = [
    "vond", "bex", "kyrn", "thess", "sarn", "dorn",
    "velthr", "muss", "gald", "phyr", "zeth", "orn",
];
/// Mixed-gender endings for nonsense epithets.
const NONSENSE_TAIL: [&str; 2] = ["us", "a"];

/// Latin epithets per signature trait, `[low, high]`.
///
/// Index order matches `species::SIGNATURE_DIMS`: vision, speed, metabolism,
/// reproduction_cost, attack, defense, aggression, intelligence, immunity. Each pool
/// deliberately mixes masculine and feminine forms.
const LATIN: [[&[&str]; 2]; SIG_LEN] = [
    // vision — low: blind/dim, high: sharp/watchful
    [&["caeca", "obscurus", "nebulosa"], &["lucida", "vigilus", "acuta", "clarus"]],
    // speed — low: slow, high: swift
    [&["tarda", "lentus", "pigra"], &["velox", "celer", "fugax", "rapida"]],
    // metabolism — low: cold/torpid, high: burning
    [&["frigida", "torpidus", "gelida"], &["ardens", "avidus", "fervida", "flagrans"]],
    // reproduction_cost — low: cheap offspring, high: costly
    [&["fecunda", "parcus", "frugalis"], &["prodiga", "sumptuosus", "gravis"]],
    // attack — low: unarmed, high: savage
    [&["mitis", "inermis", "placidus"], &["ferox", "rapax", "atrox", "acerbus"]],
    // defense — low: exposed, high: armoured
    [&["fragilis", "nuda", "apertus"], &["loricata", "munitus", "scutata", "firmus"]],
    // aggression — low: peaceable, high: warlike
    [&["tranquilla", "benignus", "quieta"], &["bellicosa", "iratus", "truculenta"]],
    // intelligence — low: dull and slow to notice, high: quick and watchful
    [&["hebes", "stolida", "obtusus", "torpens"], &["sagax", "prudens", "callida", "argutus"]],
    // immunity — low: takes every plague going, high: shrugs them off
    [&["morbida", "aegrus", "pestilens", "tabida"], &["immunis", "salubris", "incorrupta", "sanus"]],
];

/// A generated binomial.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Name {
    pub genus: String,
    pub epithet: String,
}

impl Name {
    pub fn full(&self) -> String {
        format!("{} {}", self.genus, self.epithet)
    }
}

impl std::fmt::Display for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.genus, self.epithet)
    }
}

/// Generate a name for a newly promoted species.
///
/// * `deviation` — species centroid minus population centroid, per signature dim.
/// * `inherited_genus` — the genus of the nearest existing species, when one sits
///   close enough to be the same lineage.
/// * `taken` — names already in use this run, so collisions can be redrawn.
pub fn generate(
    species_id: u32,
    world_seed: u64,
    deviation: &[f64; SIG_LEN],
    inherited_genus: Option<&str>,
    taken: &[String],
) -> Name {
    // Mixing the id in rather than adding it keeps consecutive ids from drawing
    // adjacent streams, which would make sibling species rhyme.
    let mut rng = ChaCha8Rng::seed_from_u64(
        world_seed ^ (species_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
    );

    let genus = match inherited_genus {
        Some(g) => g.to_string(),
        None => genus(&mut rng),
    };

    // Redraw on collision. Bounded: after enough tries fall through and accept,
    // which is better than looping forever on a small pool.
    for _ in 0..24 {
        let name = Name { genus: genus.clone(), epithet: epithet(&mut rng, deviation) };
        if !taken.iter().any(|t| *t == name.full()) {
            return name;
        }
    }
    Name { genus, epithet: epithet(&mut rng, deviation) }
}

/// Endings that turn a host's name into the name of what it is dying of. Real
/// pathological suffixes, so the result reads as a diagnosis rather than as
/// another animal.
const DISEASE_SUFFIX: [&str; 8] =
    ["osis", "itis", "aemia", "pestis", "tabes", "rubor", "necrosis", "morbus"];
/// Nonsense infixes. A disease is not a tidy derivation of its host — it is the
/// host's name mangled by whoever wrote it on the specimen jar.
const DISEASE_INFIX: [&str; 8] =
    ["ul", "andr", "iv", "oxy", "ther", "quil", "amb", "erg"];
/// Second word: what it does to you.
const DISEASE_EPITHET: [&str; 10] = [
    "vexans", "putrida", "livida", "maligna", "spumosa",
    "carnifex", "sordida", "gravescens", "atrata", "lenta",
];

/// Name a disease after the species it emerged in.
///
/// *Thalura ferox* becomes something like *Thalurulosis vexans*: the host's
/// genus, a nonsense infix, a pathological ending, and a second word for what
/// it does. Deterministic from `(disease_id, world_seed)` with a private RNG,
/// exactly like species naming — drawing from the world's stream would shift
/// every subsequent simulation draw.
pub fn disease_name(host_genus: &str, disease_id: u32, world_seed: u64) -> String {
    let mut rng = ChaCha8Rng::seed_from_u64(
        world_seed ^ (disease_id as u64).wrapping_mul(0xD1B5_4A32_D192_ED03),
    );

    // Trim the host genus back to a stem so the suffix lands cleanly: Thalura →
    // Thalur, Vorixa → Vorix.
    let stem = {
        let t = host_genus.trim_end_matches(['a', 'i', 'o', 'u', 'e']);
        if t.is_empty() { host_genus } else { t }
    };

    let infix = if rng.gen_bool(0.65) { *DISEASE_INFIX.choose(&mut rng).unwrap() } else { "" };
    let suffix = DISEASE_SUFFIX.choose(&mut rng).unwrap();
    let epithet = DISEASE_EPITHET.choose(&mut rng).unwrap();
    format!("{}{}{} {}", stem, infix, suffix, epithet)
}

fn genus(rng: &mut ChaCha8Rng) -> String {
    let head = GENUS_HEAD.choose(rng).unwrap();
    let mid = if rng.gen_bool(0.55) { *GENUS_MID.choose(rng).unwrap() } else { "" };
    let tail = GENUS_TAIL.choose(rng).unwrap();
    let raw = format!("{}{}{}", head, mid, tail);

    let mut c = raw.chars();
    match c.next() {
        Some(first) => first.to_uppercase().collect::<String>() + c.as_str(),
        None => raw,
    }
}

fn epithet(rng: &mut ChaCha8Rng, deviation: &[f64; SIG_LEN]) -> String {
    match dominant(deviation) {
        Some((dim, high)) => {
            let pool = LATIN[dim][usize::from(high)];
            (*pool.choose(rng).unwrap()).to_string()
        }
        None => format!(
            "{}{}",
            NONSENSE_STEM.choose(rng).unwrap(),
            NONSENSE_TAIL.choose(rng).unwrap(),
        ),
    }
}

/// The trait this species deviates from the population on most, and whether it
/// deviates upward. `None` when nothing clears `STRONG_DEVIATION` — the lineage
/// has no specialization worth naming.
pub fn dominant(deviation: &[f64; SIG_LEN]) -> Option<(usize, bool)> {
    let mut best: Option<(f64, usize, bool)> = None;
    for (d, &v) in deviation.iter().enumerate() {
        let magnitude = v.abs();
        if magnitude >= STRONG_DEVIATION && best.is_none_or(|(b, _, _)| magnitude > b) {
            best = Some((magnitude, d, v > 0.0));
        }
    }
    best.map(|(_, dim, high)| (dim, high))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat() -> [f64; SIG_LEN] {
        [0.0; SIG_LEN]
    }

    #[test]
    fn a_disease_is_named_after_its_host() {
        // Reads as a diagnosis rather than as another animal, and replays.
        let a = disease_name("Thalura", 1, 42);
        assert_eq!(a, disease_name("Thalura", 1, 42), "disease names must replay for a seed");
        assert_ne!(disease_name("Thalura", 2, 42), a, "two diseases, one name");
        assert!(a.starts_with("Thalur"), "lost the host's stem: {}", a);
        let (first, second) = a.split_once(' ').expect("expected two words");
        assert!(!second.is_empty());
        assert!(DISEASE_SUFFIX.iter().any(|e| first.ends_with(e)), "not pathologised: {}", a);
    }

    #[test]
    fn deterministic_for_id_and_seed() {
        let d = flat();
        let a = generate(1, 42, &d, None, &[]);
        let b = generate(1, 42, &d, None, &[]);
        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_give_different_names() {
        let d = flat();
        let a = generate(1, 42, &d, None, &[]);
        let b = generate(1, 99, &d, None, &[]);
        assert_ne!(a, b, "world seed must reach the names");
    }

    #[test]
    fn genus_is_always_feminine() {
        // Every genus ends in one of the feminine endings, and is capitalized.
        for id in 0..400u32 {
            let g = generate(id, 7, &flat(), None, &[]).genus;
            assert!(
                GENUS_TAIL.iter().any(|t| g.ends_with(t)),
                "non-feminine genus: {}", g,
            );
            assert!(g.chars().next().unwrap().is_uppercase(), "not capitalized: {}", g);
        }
    }

    #[test]
    fn inherited_genus_is_used_verbatim() {
        let n = generate(5, 42, &flat(), Some("Thalura"), &[]);
        assert_eq!(n.genus, "Thalura");
    }

    #[test]
    fn strong_deviation_earns_a_latin_epithet() {
        // Aggression (dim 6) far above the pond.
        let mut d = flat();
        d[6] = 0.4;
        let n = generate(1, 42, &d, None, &[]);
        assert!(
            LATIN[6][1].contains(&n.epithet.as_str()),
            "expected a high-aggression Latin epithet, got {}", n.epithet,
        );
    }

    #[test]
    fn sign_selects_the_direction() {
        let mut low = flat();
        low[1] = -0.4;               // speed far below the pond
        let n = generate(1, 42, &low, None, &[]);
        assert!(LATIN[1][0].contains(&n.epithet.as_str()), "expected a slow word, got {}", n.epithet);

        let mut high = flat();
        high[1] = 0.4;
        let n = generate(1, 42, &high, None, &[]);
        assert!(LATIN[1][1].contains(&n.epithet.as_str()), "expected a swift word, got {}", n.epithet);
    }

    #[test]
    fn no_specialization_gets_a_nonsense_epithet() {
        let mut d = flat();
        d[3] = STRONG_DEVIATION * 0.5;   // present but not strong
        let n = generate(1, 42, &d, None, &[]);
        let latin = LATIN.iter().flatten().any(|p| p.contains(&n.epithet.as_str()));
        assert!(!latin, "unspecialized species got a Latin epithet: {}", n.epithet);
        assert!(
            NONSENSE_STEM.iter().any(|s| n.epithet.starts_with(s)),
            "not a nonsense epithet: {}", n.epithet,
        );
    }

    #[test]
    fn epithets_mix_masculine_and_feminine() {
        // Across many nonsense draws both endings must appear — the convention
        // is deliberately mixed rather than agreeing with the feminine genus.
        let mut masc = false;
        let mut fem = false;
        for id in 0..200u32 {
            let e = generate(id, 3, &flat(), None, &[]).epithet;
            if e.ends_with("us") { masc = true; }
            if e.ends_with('a') { fem = true; }
        }
        assert!(masc && fem, "endings did not mix (masc={}, fem={})", masc, fem);
    }

    #[test]
    fn collisions_are_redrawn() {
        let d = flat();
        let first = generate(1, 42, &d, None, &[]);
        let second = generate(1, 42, &d, None, &[first.full()]);
        assert_ne!(first.full(), second.full(), "collision was not redrawn");
    }
}

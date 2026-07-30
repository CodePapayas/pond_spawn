// Decodes one agent's flat stride into a plain object. Single place that
// knows the buffer offsets — everything downstream works with named fields.

// Field offsets within an agent's stride. Must match wasm.rs A_* constants.
const A_X = 0;
const A_Y = 1;
const A_PREV_X = 2;
const A_PREV_Y = 3;
const A_ENERGY_NORM = 4;
const A_VEL_X = 5;
const A_VEL_Y = 6;
const A_GENOME_CLUSTER = 7;
const A_BRAIN_CLUSTER = 8;
const A_AGE_NORM = 9;
const A_ID = 10;
const A_MORPH_POINTINESS = 11;
const A_MORPH_ELONGATION = 12;
const A_MORPH_BULK = 13;
const A_MORPH_ORNAMENT = 14;
const A_MORPH_EYE_SIZE = 15;
const A_MORPH_PULSE_RATE = 16;
const A_MORPH_BELLY = 17;
const A_SPECIES = 18;

/** Decode agent slot `i` (0-based) out of `buf`, given the stride layout. */
export function decodeAgent(buf, i, headerLen, agentStride) {
    const off = headerLen + i * agentStride;
    return {
        x: buf[off + A_X],
        y: buf[off + A_Y],
        prevX: buf[off + A_PREV_X],
        prevY: buf[off + A_PREV_Y],
        energyNorm: buf[off + A_ENERGY_NORM],
        velX: buf[off + A_VEL_X],
        velY: buf[off + A_VEL_Y],
        genomeCluster: buf[off + A_GENOME_CLUSTER] | 0,
        brainCluster: buf[off + A_BRAIN_CLUSTER] | 0,
        ageNorm: buf[off + A_AGE_NORM],
        id: buf[off + A_ID] | 0,
        species: buf[off + A_SPECIES] | 0,
        morph: {
            pointiness: buf[off + A_MORPH_POINTINESS],
            elongation: buf[off + A_MORPH_ELONGATION],
            bulk: buf[off + A_MORPH_BULK],
            ornament: buf[off + A_MORPH_ORNAMENT],
            eyeSize: buf[off + A_MORPH_EYE_SIZE],
            pulseRate: buf[off + A_MORPH_PULSE_RATE],
            belly: buf[off + A_MORPH_BELLY],
        },
    };
}

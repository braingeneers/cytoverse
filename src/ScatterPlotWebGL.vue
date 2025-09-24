<template>
  <div ref="containerRef" class="scatterplot-container">
    <canvas ref="canvasRef" class="scatterplot-canvas" />
    <div class="legend">
      <div class="legend-item">
        <div class="legend-dot base-ref"></div>
        <span class="legend-label">Base Reference</span>
      </div>
      <div class="legend-item">
        <div class="legend-dot user-ref"></div>
        <span class="legend-label">User Reference</span>
      </div>
      <div class="legend-item">
        <div class="legend-dot query"></div>
        <span class="legend-label">Query</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * High performance scatter plot that displays:
 * BaseReference - base underlying reference dataset i.e SCimilarity training data
 * UserReference - optional user reference based on a former query against this base ref
 * Query - current query set of cells being labeled against either the base or user
 *
 * NOTE: We DEPEND that the x/y Base Ref points are the first N reference cells and
 * correspond to the first N baseRefLabelIndices. The overall base reference
 * needs the indices for every cell in the base reference to do labeling, but for
 * plotting we only show the first N (see pumap_train.py)
 */
import { ref, onMounted, onUnmounted, watch } from 'vue';
import createScatterplot from 'regl-scatterplot';

// Props interface
interface Props {
  baseRefX: Float32Array;
  baseRefY: Float32Array;
  baseRefLabelIndices: Int16Array;
  baseRefLabels: string[];
  userRefX: Float32Array;
  userRefY: Float32Array;
  userRefLabelIndices: Int16Array;
  queryX: Float32Array;
  queryY: Float32Array;
  queryLabelIndices: Int16Array;
}

// Point size for base reference, user reference, query respectively
const POINT_SIZES = [1, 3, 6];

const props = defineProps<Props>();

// Template refs
const containerRef = ref<HTMLDivElement>();
const canvasRef = ref<HTMLCanvasElement>();

// Component state
let scatterplotRef: ReturnType<typeof createScatterplot> | null = null;
const isInitializedRef = ref(false);
const isDrawingRef = ref(false);
let updateInterval: number | null = null;

// Generate distinct colors for labels + one extra color for test points
// This function generates a set of colors that are evenly spaced in the HSL color space,
// avoiding bright red (hue 0) to ensure good visibility and distinction between labels.
// It returns an array of hex color strings.
// The number of labels is passed as an argument, and it generates colors starting from hue 30 (yellow) to hue 360 (red), ensuring that the colors are visually distinct and not too bright or saturated.
// The colors are generated in HSL format and converted to hex format for use in the scatterplot.
const generateLabelColors = (numLabels: number): string[] => {
  const colors: string[] = [];

  // Use HSL to generate evenly spaced colors, avoiding bright red (hue 0)
  for (let i = 0; i < numLabels; i++) {
    // Skip hue 0 (red) and start from 30 degrees to avoid bright red
    const hue = 30 + (i * 330) / numLabels;
    const saturation = 0.7;
    const lightness = 0.5;

    // Convert HSL to RGB
    const c = (1 - Math.abs(2 * lightness - 1)) * saturation;
    const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
    const m = lightness - c / 2;

    let r: number, g: number, b: number;

    if (hue >= 0 && hue < 60) {
      r = c;
      g = x;
      b = 0;
    } else if (hue >= 60 && hue < 120) {
      r = x;
      g = c;
      b = 0;
    } else if (hue >= 120 && hue < 180) {
      r = 0;
      g = c;
      b = x;
    } else if (hue >= 180 && hue < 240) {
      r = 0;
      g = x;
      b = c;
    } else if (hue >= 240 && hue < 300) {
      r = x;
      g = 0;
      b = c;
    } else {
      r = c;
      g = 0;
      b = x;
    }

    const rHex = Math.round((r + m) * 255)
      .toString(16)
      .padStart(2, '0');
    const gHex = Math.round((g + m) * 255)
      .toString(16)
      .padStart(2, '0');
    const bHex = Math.round((b + m) * 255)
      .toString(16)
      .padStart(2, '0');

    colors.push(`#${rHex}${gHex}${bHex}`);
  }

  colors.push('#999999'); // Add gray for un-labeled query cells

  return colors;
};

// Handle resize
const handleResize = () => {
  if (containerRef.value && scatterplotRef) {
    scatterplotRef.set({
      width: containerRef.value.clientWidth,
      height: containerRef.value.clientHeight,
    });
  }
};

// Initial setup and reference rendering
const initializeScatterplot = () => {
  if (
    !containerRef.value ||
    !canvasRef.value ||
    !props.baseRefX ||
    !props.baseRefY ||
    !props.baseRefLabelIndices
  )
    return;

  // Generate colors for each category + one extra color for test points
  const labelColors = generateLabelColors(props.baseRefLabels.length);

  // Create the scatterplot instance
  const scatterplot = createScatterplot({
    canvas: canvasRef.value,
    width: containerRef.value.clientWidth,
    height: containerRef.value.clientHeight,
    // Use a range for point sizes: base reference, user reference, query
    pointSize: POINT_SIZES,
    // performanceMode: true, // Enable performance mode for better rendering
  });

  // Set up label coloring and sizing
  scatterplot.set({
    pointColor: labelColors,
    colorBy: 'valueA',
    sizeBy: 'valueB',
  });

  scatterplotRef = scatterplot;

  const initialColumnData = {
    x: new Float32Array([...props.baseRefX, ...props.userRefX]),
    y: new Float32Array([...props.baseRefY, ...props.userRefY]),
    valueA: [...props.baseRefLabelIndices, ...props.userRefLabelIndices],
    valueB: [
      ...new Array(props.baseRefX.length).fill(0),
      ...new Array(props.userRefX.length).fill(1),
    ],
  };

  // Draw the initial reference points
  const drawInitialData = async () => {
    await scatterplot.draw(initialColumnData);
    isInitializedRef.value = true;
    console.log(
      'Drew',
      props.baseRefX.length + props.userRefX.length,
      'points with',
      labelColors.length,
      'label colors'
    );
  };
  drawInitialData();
};

const updateColumnData = () => {
  if (!props.baseRefX || !props.baseRefY || !props.baseRefLabelIndices) {
    return null;
  }
  return {
    x: new Float32Array([...props.baseRefX, ...props.userRefX, ...props.queryX]),
    y: new Float32Array([...props.baseRefY, ...props.userRefY, ...props.queryY]),
    valueA: [
      ...props.baseRefLabelIndices,
      ...props.userRefLabelIndices,
      ...props.queryLabelIndices,
    ],
    valueB: [
      ...new Array(props.baseRefX.length).fill(0),
      ...new Array(props.userRefX.length).fill(1),
      ...new Array(props.queryX.length).fill(2),
    ],
  };
};

// Timer-based update function
const doTimerUpdate = async () => {
  if (!scatterplotRef || !isInitializedRef.value || isDrawingRef.value) {
    return;
  }

  const columnData = updateColumnData();
  if (!columnData) return;

  isDrawingRef.value = true;
  try {
    await scatterplotRef.draw(columnData);
  } finally {
    isDrawingRef.value = false;
  }
};

// Start/stop timer updates
const startTimerUpdates = () => {
  if (updateInterval) return;
  updateInterval = window.setInterval(doTimerUpdate, 1000);
};

const stopTimerUpdates = () => {
  if (updateInterval) {
    clearInterval(updateInterval);
    updateInterval = null;
  }
};

// Force immediate update (for final refresh)
const forceUpdate = () => {
  doTimerUpdate();
};

// Watch only reference data changes (need to reinitialize)
watch(
  [
    () => props.baseRefX,
    () => props.baseRefY,
    () => props.baseRefLabelIndices,
    () => props.baseRefLabels,
  ],
  () => {
    stopTimerUpdates();
    if (scatterplotRef) {
      scatterplotRef.destroy();
      scatterplotRef = null;
      isInitializedRef.value = false;
    }
    initializeScatterplot();
  }
);

// Expose control functions
defineExpose({
  startTimerUpdates,
  stopTimerUpdates,
  forceUpdate,
});

onMounted(() => {
  initializeScatterplot();
  window.addEventListener('resize', handleResize);
});

onUnmounted(() => {
  window.removeEventListener('resize', handleResize);
  stopTimerUpdates();
  if (scatterplotRef) {
    scatterplotRef.destroy();
  }
});
</script>

<style scoped>
.scatterplot-container {
  width: 100%;
  height: 100%;
  overflow: hidden;
  position: relative;
}

.scatterplot-canvas {
  width: 100%;
  height: 100%;
  display: block;
}

.legend {
  position: absolute;
  top: 10px;
  left: 10px;
  z-index: 10;
  pointer-events: none;
}

.legend-item {
  display: flex;
  align-items: center;
  margin-bottom: 6px;
}

.legend-item:last-child {
  margin-bottom: 0;
}

.legend-dot {
  width: 12px;
  height: 12px;
  margin-right: 8px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}

.legend-dot.base-ref::after {
  content: '';
  width: 2px;
  height: 2px;
  border-radius: 50%;
  background-color: #1976d2;
  display: block;
}

.legend-dot.user-ref::after {
  content: '';
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background-color: #1976d2;
  display: block;
}

.legend-dot.query::after {
  content: '';
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: #1976d2;
  display: block;
}

.legend-label {
  font-size: 12px;
  color: white;
}
</style>

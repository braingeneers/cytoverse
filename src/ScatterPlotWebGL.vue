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
    <!-- Cell Info Popup -->
    <div
      v-if="selectedCell"
      class="cell-info-popup"
      :style="{
        left: `${popupPosition.x}px`,
        top: `${popupPosition.y}px`,
      }"
    >
      <div class="popup-header">
        <span class="popup-title">Cell Information</span>
        <button class="popup-close" @click="closePopup">×</button>
      </div>
      <div class="popup-content">
        <div class="popup-row">
          <span class="popup-label">Cell ID:</span>
          <span class="popup-value">{{ selectedCell.cellId }}</span>
        </div>
        <div class="popup-row">
          <span class="popup-label">Label:</span>
          <span class="popup-value">{{ selectedCell.label }}</span>
        </div>
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
import { generateLabelColors } from './colors';

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
  queryCellNames?: string[];
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

// Popup state
const selectedCell = ref<{ cellId: string; label: string } | null>(null);
const popupPosition = ref({ x: 0, y: 0 });

// Handle resize
const handleResize = () => {
  if (containerRef.value && scatterplotRef) {
    scatterplotRef.set({
      width: containerRef.value.clientWidth,
      height: containerRef.value.clientHeight,
    });
  }
};

// Handle cell click
const handleCellClick = (event: { points: number[] }) => {
  if (!event.points || event.points.length === 0) {
    closePopup();
    return;
  }

  const pointIndex = event.points[0];
  const baseRefCount = props.baseRefX.length;
  const userRefCount = props.userRefX.length;
  const queryStartIndex = baseRefCount + userRefCount;

  let cellId: string;
  let label: string;

  if (pointIndex < baseRefCount) {
    // Base reference cell
    cellId = `Base Ref ${pointIndex}`;
    const labelIndex = props.baseRefLabelIndices[pointIndex];
    label =
      labelIndex >= 0 && labelIndex < props.baseRefLabels.length
        ? props.baseRefLabels[labelIndex]
        : 'Unknown';
  } else if (pointIndex < queryStartIndex) {
    // User reference cell
    const userRefIndex = pointIndex - baseRefCount;
    cellId = `User Ref ${userRefIndex}`;
    const labelIndex = props.userRefLabelIndices[userRefIndex];
    label =
      labelIndex >= 0 && labelIndex < props.baseRefLabels.length
        ? props.baseRefLabels[labelIndex]
        : 'Unknown';
  } else {
    // Query cell
    const queryCellIndex = pointIndex - queryStartIndex;
    if (queryCellIndex < props.queryX.length) {
      cellId = props.queryCellNames?.[queryCellIndex] || `Query ${queryCellIndex}`;
      const labelIndex = props.queryLabelIndices[queryCellIndex];
      label =
        labelIndex >= 0 && labelIndex < props.baseRefLabels.length
          ? props.baseRefLabels[labelIndex]
          : 'Unknown';
    } else {
      closePopup();
      return;
    }
  }

  selectedCell.value = { cellId, label };

  // Position popup near the mouse cursor
  // We'll use a simple offset from the canvas center for now
  if (containerRef.value) {
    popupPosition.value = {
      x: containerRef.value.clientWidth / 2 - 150,
      y: 50,
    };
  }
};

// Close popup
const closePopup = () => {
  selectedCell.value = null;
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

  // Subscribe to click events
  scatterplot.subscribe('select', handleCellClick);

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

.cell-info-popup {
  position: absolute;
  z-index: 20;
  background-color: rgba(30, 30, 30, 0.95);
  border: 1px solid #424242;
  border-radius: 6px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
  min-width: 300px;
  max-width: 400px;
  pointer-events: auto;
}

.popup-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid #424242;
  background-color: rgba(25, 25, 25, 0.95);
  border-radius: 6px 6px 0 0;
}

.popup-title {
  font-size: 14px;
  font-weight: 600;
  color: white;
}

.popup-close {
  background: none;
  border: none;
  color: #999;
  font-size: 24px;
  line-height: 1;
  cursor: pointer;
  padding: 0;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: color 0.2s;
}

.popup-close:hover {
  color: white;
}

.popup-content {
  padding: 12px 16px;
}

.popup-row {
  display: flex;
  gap: 12px;
  margin-bottom: 8px;
  font-size: 13px;
}

.popup-row:last-child {
  margin-bottom: 0;
}

.popup-label {
  font-weight: 600;
  color: #999;
  min-width: 60px;
}

.popup-value {
  color: white;
  word-break: break-all;
}
</style>

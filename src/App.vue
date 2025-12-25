<template>
  <v-app>
    <v-layout>
      <!-- Mini Sidebar - Always visible when main sidebar is closed -->
      <div v-if="!sidebarOpen && !isMobile" class="mini-sidebar">
        <div class="mini-toolbar">
          <v-btn color="primary" icon size="large" @click="handleDrawerToggle">
            <v-icon>mdi-menu</v-icon>
          </v-btn>
        </div>
        <div class="mini-actions">
          <v-btn
            color="primary"
            icon
            size="large"
            :disabled="!selectedFile || isLoadingData"
            @click="handleRunStopClick"
          >
            <v-icon>{{ isRunning ? 'mdi-stop' : 'mdi-play' }}</v-icon>
          </v-btn>
        </div>
      </div>

      <!-- Main Sidebar -->
      <v-navigation-drawer
        v-model="sidebarOpen"
        :location="isMobile ? 'right' : 'left'"
        :temporary="isMobile"
        :width="drawerWidth"
        class="main-drawer"
      >
        <div class="drawer-header">
          <h2>Cytoverse</h2>
          <div class="header-actions">
            <v-btn variant="text" icon size="small" @click="helpModalOpen = true">
              <v-icon>mdi-help-circle</v-icon>
            </v-btn>
            <v-btn variant="text" icon size="small" @click="openGithub">
              <v-icon>mdi-github</v-icon>
            </v-btn>
            <v-btn variant="text" icon @click="handleDrawerToggle">
              <v-icon>mdi-arrow-left</v-icon>
            </v-btn>
          </div>
        </div>

        <div class="drawer-content">
          <!-- File Selection -->
          <div class="form-section">
            <div style="display: flex; gap: 8px; align-items: stretch">
              <v-file-input
                :model-value="selectedFile ? [selectedFile] : []"
                accept=".h5ad"
                label="Query"
                placeholder="Select an .h5ad file"
                prepend-icon=""
                variant="outlined"
                density="comfortable"
                style="flex: 1"
                @update:model-value="handleFileSelect"
              />

              <v-btn
                variant="text"
                icon
                size="small"
                title="Select .h5ad Query File"
                style="width: 40px; min-width: 40px; height: 40px"
                @click="openFileDialog"
              >
                <v-icon>mdi-folder-open</v-icon>
              </v-btn>
            </div>
          </div>

          <!-- Model Selection -->
          <div class="form-section">
            <div style="display: flex; gap: 8px; align-items: stretch">
              <v-select
                v-model="selectedModel"
                data-testid="model-select-dropdown"
                :items="availableModels"
                label="Reference"
                :disabled="isRunning"
                variant="outlined"
                density="comfortable"
                style="flex: 1"
              >
                <template #item="{ item, props }">
                  <v-list-item v-bind="props">
                    <template v-if="item.raw.isUserIndex" #append>
                      <v-btn
                        icon
                        size="x-small"
                        variant="text"
                        title="Delete User Reference"
                        @click.stop.prevent="
                          confirmDeleteUserIndex(item.raw.value, item.raw.title)
                        "
                      >
                        <v-icon size="small">mdi-delete</v-icon>
                      </v-btn>
                    </template>
                  </v-list-item>
                </template>
              </v-select>
              <v-btn
                variant="text"
                icon
                size="small"
                title="Import Reference"
                data-testid="import-index-button"
                style="width: 40px; min-width: 40px; height: 40px"
                @click="handleImportIndex"
              >
                <v-icon>mdi-folder-open</v-icon>
              </v-btn>
            </div>
          </div>

          <!-- Category Selection -->
          <div class="form-section">
            <v-select
              v-model="selectedCategory"
              data-testid="category-select-dropdown"
              :items="availableCategories"
              label="Category"
              :disabled="isRunning"
              variant="outlined"
              density="comfortable"
            />
          </div>

          <!-- Dataset Statistics -->
          <v-card class="stats-card" variant="outlined">
            <v-card-text>
              <template
                v-if="baseRef.x && baseRef.y && baseRef.labels && baseRef.labelIndices"
              >
                <div class="stat-item">
                  <strong>Reference Cells:</strong>
                  {{ baseRef.labelIndices.value.length.toLocaleString() }}
                </div>
                <div class="stat-item">
                  <strong>Plotted Cells:</strong>
                  {{ baseRef.x.value.length.toLocaleString() }}
                </div>
                <div class="stat-item">
                  <strong>Labels:</strong> {{ baseRef.labels.value.length }}
                </div>
              </template>
              <template v-else-if="isLoadingData">
                <div class="loading-text">Loading statistics...</div>
              </template>
              <template v-else>
                <div class="loading-text">No data loaded</div>
              </template>
            </v-card-text>
          </v-card>

          <!-- WebGPU Selection -->
          <div class="form-section">
            <v-radio-group v-model="useWebGPU" :disabled="isRunning" inline>
              <v-radio :value="false" label="CPU" data-testid="radio-cpu-option" />
              <v-radio
                :value="true"
                :disabled="!hasWebGPU"
                data-testid="radio-gpu-option"
              >
                <template #label>
                  <span>GPU</span>
                  <v-tooltip
                    v-if="!hasWebGPU"
                    text="WebGPU not available in this browser"
                  >
                    <template #activator="{ props }">
                      <v-icon v-bind="props" size="small" class="ml-1"
                        >mdi-information</v-icon
                      >
                    </template>
                  </v-tooltip>
                </template>
              </v-radio>
            </v-radio-group>
          </div>

          <!-- Run/Stop Button -->
          <v-btn
            color="primary"
            size="large"
            :prepend-icon="isRunning ? 'mdi-stop' : 'mdi-play'"
            :disabled="!selectedFile || isLoadingData"
            data-testid="run-stop-button"
            block
            @click="handleRunStopClick"
          >
            {{ isRunning ? 'Stop' : 'Start' }}
          </v-btn>

          <!-- Create Index Button -->
          <v-btn
            v-if="canCreateIndex"
            color="primary"
            size="large"
            prepend-icon="mdi-database-plus"
            data-testid="create-index-button"
            block
            class="mt-1"
            @click="createIndexModalOpen = true"
          >
            Create Reference
          </v-btn>

          <!-- Share User Reference Button -->
          <v-btn
            v-if="selectedModelInfo?.isUserIndex"
            color="primary"
            size="large"
            prepend-icon="mdi-share"
            data-testid="share-reference-button"
            block
            class="mt-1"
            @click="handleShareExportForSelected"
          >
            Share User Reference
          </v-btn>

          <!-- Progress and Status -->
          <div v-if="progress > 0 || statusMessage" class="progress-section">
            <div class="status-text" data-testid="status">
              {{ statusMessage }}
            </div>
            <v-progress-linear :model-value="progress" color="primary" />
            <div class="progress-text">{{ progress.toFixed(1) }}%</div>
          </div>

          <!-- Label Counts -->
          <div v-if="Object.keys(labelCounts).length > 0" class="labels-section">
            <div class="labels-header">
              <h3>Labels ({{ selectedCategory }})</h3>
              <div class="header-actions">
                <v-btn
                  variant="text"
                  icon
                  size="small"
                  :disabled="Object.keys(labelCounts).length === 0 || isRunning"
                  data-testid="download-button"
                  title="Download results as CSV"
                  @click="exportResultsToCSV"
                >
                  <v-icon>mdi-download</v-icon>
                </v-btn>
              </div>
            </div>

            <div class="total-count">
              <span>Total:</span>
              <span>{{ totalLabeled.toLocaleString() }}</span>
            </div>

            <div
              v-for="[label, count] in sortedLabelCounts"
              :key="label"
              class="label-count"
            >
              <span class="label-name">
                <span
                  class="color-dot"
                  :style="{ backgroundColor: labelColorMap.get(label) }"
                ></span>
                {{ label }}
              </span>
              <span>{{ count.toLocaleString() }}</span>
            </div>
          </div>
        </div>
      </v-navigation-drawer>

      <!-- Main Content -->
      <v-main>
        <div
          class="main-content"
          :style="{
            marginLeft: sidebarOpen ? 0 : isMobile ? 0 : miniDrawerWidth + 'px',
          }"
        >
          <div v-if="isLoadingData" class="loading-container">
            <div>Loading scatter plot data...</div>
          </div>
          <ScatterPlotWebGL
            v-else-if="
              baseRef.x.value &&
              baseRef.y.value &&
              baseRef.labelIndices.value &&
              baseRef.labels.value.length > 0
            "
            ref="scatterPlotRef"
            :base-ref-x="baseRef.x.value"
            :base-ref-y="baseRef.y.value"
            :base-ref-label-indices="baseRef.labelIndicesSubset.value"
            :base-ref-labels="baseRef.labels.value"
            :user-ref-x="userRef.x.value"
            :user-ref-y="userRef.y.value"
            :user-ref-label-indices="userRef.labelIndices.value"
            :query-x="query.x.value"
            :query-y="query.y.value"
            :query-label-indices="query.labelIndices.value"
            :query-cell-names="query.cellNames.value"
          />
          <div v-else class="loading-container">
            <div>No data available</div>
          </div>
        </div>
      </v-main>

      <!-- Share Modal -->
      <v-dialog v-model="shareModalOpen" max-width="500px">
        <v-card>
          <v-card-title>Share User Reference</v-card-title>
          <v-card-text>
            <div class="modal-description">
              Export this user reference as a JSON file that can be shared with others.
              The exported file contains all the embeddings and labels from your
              analysis.
            </div>
          </v-card-text>
          <v-card-actions>
            <v-spacer />
            <v-btn @click="shareModalOpen = false">Cancel</v-btn>
            <v-btn color="primary" @click="handleShareExport"> Export </v-btn>
          </v-card-actions>
        </v-card>
      </v-dialog>

      <!-- Help Modal -->
      <v-dialog v-model="helpModalOpen" max-width="600px">
        <v-card>
          <v-card-title>Help</v-card-title>
          <v-card-text>
            <div class="help-content">
              <p>
                CytoVerse is a browser-based platform for single-cell RNA-seq analysis,
                designed for cell annotation using foundation model embeddings. It runs
                entirely in the browser, streaming h5ad files from local storage without
                uploading data or requiring server computation. It uses SCimilarity for
                cell embeddings and parametric UMAP for 2D visualization, leveraging
                large training datasets for accurate cell annotation. An Inverted File
                with Product Quantization (IVFPQ) enables fast approximate nearest
                neighbor searches across over 20 million samples. Built on WebAssembly
                and ONNX for high-speed processing, it supports unlimited streaming
                analysis of h5ad files via h5wasm. This enables distributed
                collaborative discovery, allowing researchers to explore shared
                embedding spaces to identify overlapping or complementary assays,
                particularly for perturbseq-driven research. The architecture ensures
                privacy, scalability, and collaborative potential without server
                dependency.
              </p>
              <p><strong>Getting Started:</strong></p>
              <ol>
                <li>Select an H5AD file containing single-cell RNA-seq data</li>
                <li>Choose a pre-trained model (Brain or Scimilarity)</li>
                <li>Select a category for cell type prediction</li>
                <li>Choose CPU or GPU processing (if available)</li>
                <li>Click Start to begin analysis</li>
                <li>Optionally create a user reference and share</li>
              </ol>
              <p>
                The tool will display your cells in real-time as they are processed and
                labeled.
              </p>
            </div>
          </v-card-text>
          <v-card-actions>
            <v-spacer />
            <v-btn @click="helpModalOpen = false">Close</v-btn>
          </v-card-actions>
        </v-card>
      </v-dialog>

      <!-- Error Modal -->
      <v-dialog v-model="errorModalOpen" max-width="500px">
        <v-card>
          <v-card-title>Error</v-card-title>
          <v-card-text>
            <div data-testid="error-title">{{ errorMessage }}</div>
          </v-card-text>
          <v-card-actions>
            <v-spacer />
            <v-btn @click="errorModalOpen = false">Close</v-btn>
          </v-card-actions>
        </v-card>
      </v-dialog>

      <!-- Create User Reference Modal -->
      <v-dialog v-model="createIndexModalOpen" max-width="500px">
        <v-card>
          <v-card-title>Create User Reference</v-card-title>
          <v-card-text>
            <p>
              Save the current labeled cells as a user reference for future labeling.
            </p>
            <v-text-field
              v-model="newIndexName"
              label="Reference Name"
              placeholder="Enter a name for your reference"
              data-testid="new-index-name"
              variant="outlined"
              density="comfortable"
              class="mt-4"
            />
          </v-card-text>
          <v-card-actions>
            <v-spacer />
            <v-btn @click="createIndexModalOpen = false">Cancel</v-btn>
            <v-btn
              color="primary"
              :disabled="!newIndexName.trim()"
              data-testid="save-index-button"
              @click="saveUserIndex"
            >
              Save Index
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-dialog>

      <!-- Delete Index Confirmation Modal -->
      <v-dialog v-model="deleteIndexModalOpen" max-width="400px">
        <v-card>
          <v-card-title>Delete User Index</v-card-title>
          <v-card-text>
            Are you sure you want to delete the user index "{{
              indexToDelete?.name?.replace('👤 ', '')
            }}"?
          </v-card-text>
          <v-card-actions>
            <v-spacer />
            <v-btn @click="deleteIndexModalOpen = false">Cancel</v-btn>
            <v-btn color="error" @click="deleteUserIndex"> Delete </v-btn>
          </v-card-actions>
        </v-card>
      </v-dialog>
    </v-layout>
  </v-app>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue';

import Worker from './worker?worker';
import ScatterPlotWebGL from './ScatterPlotWebGL.vue';
import { userIndexService } from './userIndexService';
import { generateLabelColors } from './colors';

// Constants
const drawerWidth = 320;
const miniDrawerWidth = 64;

// Types for cell updates
interface CellUpdate {
  cellId: string;
  x: number;
  y: number;
  label?: string;
  labelId?: number;
  confidence?: number;
  trainVectorId?: number;
}

interface CellBatchUpdate {
  type: 'cell_batch_update';
  cells: CellUpdate[];
}

// Reactive state
const sidebarOpen = ref(true);
const selectedFile = ref<File | null>(null);
const fileList = ref<File[]>([]);
const statusMessage = ref('');
const progress = ref(0);
const selectedModel = ref('scimilarity');
const availableModels = ref<
  {
    title: string;
    value: string;
    isUserIndex: boolean;
    baseModelId?: string;
    cellCount?: number;
    category?: string;
  }[]
>([]);
const isRunning = ref(false);
const hasWebGPU = ref(false);
const useWebGPU = ref(false);
const maxTextureSize = ref(0);
const isMobile = ref(window.innerWidth < 768);
const isLoadingData = ref(false);

const availableCategories = ref<string[]>([]);
const selectedCategory = ref('');

// Base reference
const baseRef = {
  labels: ref<string[]>([]),
  labelIndices: ref<Int16Array>(new Int16Array()),
  // Subset for plotted cells so we can append query
  labelIndicesSubset: ref<Int16Array>(new Int16Array()),
  x: ref<Float32Array>(new Float32Array()),
  y: ref<Float32Array>(new Float32Array()),
};

// User reference based on base reference
const userRef = {
  x: ref<Float32Array>(new Float32Array()),
  y: ref<Float32Array>(new Float32Array()),
  labelIndices: ref<Int16Array>(new Int16Array()),
};

// Query
const query = {
  x: ref<Float32Array>(new Float32Array()),
  y: ref<Float32Array>(new Float32Array()),
  labelIndices: ref<Int16Array>(new Int16Array()),
  confidences: ref<Float32Array>(new Float32Array()),
  cellNames: ref<string[]>([]),
};

// Processing state
const totalNumCells = ref(0);
const totalProcessed = ref(0);
const startTime = ref<number | null>(null);

// Labeling feedback state
const labelCounts = ref<{ [label: string]: number }>({});
const totalLabeled = ref(0);

// Modal states
const shareModalOpen = ref(false);
const shareIndexInfo = ref<{ id: string; name: string } | null>(null);
const helpModalOpen = ref(false);
const errorModalOpen = ref(false);
const errorMessage = ref('');
const createIndexModalOpen = ref(false);
const newIndexName = ref('');
const deleteIndexModalOpen = ref(false);
const indexToDelete = ref<{ id: string; name: string } | null>(null);

// User Index state
const canCreateIndex = ref(false);
const currentUserIndexArtifacts = ref<{
  partitionIds: Int16Array;
  pqCodes: Uint8Array[];
} | null>(null);

// Normalization parameters from metadata
const xCenter = ref(0);
const yCenter = ref(0);
const maxRange = ref(1);

// Worker management - single worker
let worker: Worker | null = null;
const cellPositions = new Map<string, { x: number; y: number }>(); // Track cell positions

// ScatterPlot ref
const scatterPlotRef = ref<{
  startTimerUpdates: () => void;
  stopTimerUpdates: () => void;
  forceUpdate: () => void;
} | null>(null);

// Site path calculation
const sitePath =
  window.location.origin +
  window.location.pathname.slice(0, window.location.pathname.lastIndexOf('/'));

// Computed properties
const sortedLabelCounts = computed(() => {
  return Object.entries(labelCounts.value).sort(([, a], [, b]) => b - a);
});

const selectedModelInfo = computed(() => {
  return availableModels.value.find((m) => m.value === selectedModel.value);
});

// Generate label colors that match the scatter plot
const labelColorMap = computed(() => {
  if (baseRef.labels.value.length === 0) return new Map<string, string>();
  const colors = generateLabelColors(baseRef.labels.value.length);
  const colorMap = new Map<string, string>();
  baseRef.labels.value.forEach((label, index) => {
    colorMap.set(label, colors[index]);
  });
  return colorMap;
});

// Helper function to normalize coordinates
const normalizeCoordinates = (x: number, y: number) => {
  const normalizedX = (x - xCenter.value) / (maxRange.value / 2);
  const normalizedY = (y - yCenter.value) / (maxRange.value / 2);
  return [normalizedX, normalizedY];
};

// Track cell ID to index mapping
const cellIdToIndex = new Map<string, number>();

// Handle cell updates from worker
const handleCellBatchUpdate = (batchUpdate: CellBatchUpdate) => {
  // Process all cells in batch
  const newXData: number[] = [];
  const newYData: number[] = [];
  const newLabels: number[] = [];
  const updatedIndices: number[] = [];

  for (const update of batchUpdate.cells) {
    // Normalize coordinates
    const [normalizedX, normalizedY] = normalizeCoordinates(update.x, update.y);

    let cellIndex = cellIdToIndex.get(update.cellId);

    if (cellIndex === undefined) {
      // New cell - add to arrays
      cellIndex = query.x.value.length + newXData.length;
      cellIdToIndex.set(update.cellId, cellIndex);

      newXData.push(normalizedX);
      newYData.push(normalizedY);
      newLabels.push(update.labelId ?? -1);
    } else {
      // Update existing cell
      updatedIndices.push(cellIndex);
      query.x.value[cellIndex] = normalizedX;
      query.y.value[cellIndex] = normalizedY;
      if (update.labelId !== undefined) {
        query.labelIndices.value[cellIndex] = update.labelId;
      }
    }

    // Update label counts if labeled
    if (
      update.labelId !== undefined &&
      update.labelId >= 0 &&
      update.labelId < baseRef.labels.value.length
    ) {
      const label = baseRef.labels.value[update.labelId];
      labelCounts.value[label] = (labelCounts.value[label] || 0) + 1;
      totalLabeled.value += 1;
    }
  }

  // Batch append new data to arrays
  // REMIND: Switch to set() as its more efficient
  if (newXData.length > 0) {
    query.x.value = new Float32Array([...query.x.value, ...newXData]);
    query.y.value = new Float32Array([...query.y.value, ...newYData]);
    query.labelIndices.value = new Int16Array([
      ...query.labelIndices.value,
      ...newLabels,
    ]);
    query.confidences.value = new Float32Array([
      ...query.confidences.value,
      ...batchUpdate.cells.map((c) => c.confidence ?? -1),
    ]);
  }
};

// Terminate worker
const terminateWorker = () => {
  if (worker) {
    console.log('Terminating worker...');
    worker.terminate();
    worker = null;
  }
};
// Create worker
const createWorker = () => {
  console.log('Creating worker...');

  worker = new Worker();

  worker.onmessage = (evt) => {
    switch (evt.data.type) {
      case 'status':
        statusMessage.value = evt.data.message;
        break;
      case 'progress':
        progress.value = Math.min(
          100,
          Math.round((evt.data.countFinished / evt.data.totalToProcess) * 100)
        );
        totalProcessed.value = evt.data.countFinished;
        statusMessage.value = `${
          evt.data.message
        } ${evt.data.countFinished.toLocaleString()} of ${evt.data.totalToProcess.toLocaleString()}...`;
        break;
      case 'cell_batch_update':
        handleCellBatchUpdate(evt.data as CellBatchUpdate);
        break;
      case 'finished':
        console.log('Worker finished processing');
        isRunning.value = false;
        scatterPlotRef.value?.stopTimerUpdates();
        scatterPlotRef.value?.forceUpdate();

        // Update cell names
        query.cellNames.value = evt.data.cellNames;
        console.log('Cell Names:', query.cellNames.value.slice(0, 10));

        // Store user index artifacts if available and we're using a reference model
        if (evt.data.partitionIds && evt.data.pqCodes) {
          const selectedModelInfo = availableModels.value.find(
            (m) => m.value === selectedModel.value
          );
          const isUsingReferenceModel = !selectedModelInfo?.isUserIndex;
          currentUserIndexArtifacts.value = {
            partitionIds: evt.data.partitionIds,
            pqCodes: evt.data.pqCodes,
          };
          // currentUserIndexArtifacts.value?.pqCodes = evt.data.pqCodes;
          canCreateIndex.value = isUsingReferenceModel; // Only allow creating from reference models
        }

        if (startTime.value) {
          const endTime = Date.now();
          const totalElapsed = Math.round((endTime - startTime.value) / 1000);
          const minutes = Math.floor(totalElapsed / 60);
          const seconds = totalElapsed % 60;
          const timeStr = minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
          statusMessage.value = `Finished: Labeled ${totalProcessed.value.toLocaleString()} cells in ${timeStr}`;
        } else {
          statusMessage.value = 'Finished';
        }
        break;
      case 'error':
        console.error('Worker error:', evt.data.error);
        isRunning.value = false;
        statusMessage.value = '';
        errorMessage.value = evt.data.error.toString();
        errorModalOpen.value = true;
        break;
    }
  };
};

const loadAvailableModels = async () => {
  try {
    // Load reference models
    const response = await fetch('models/models.txt');
    const text = await response.text();
    const modelNames = text.split('\n').filter((name) => name.trim());

    const referenceModels = modelNames.map((name) => ({
      title: name,
      value: name,
      isUserIndex: false,
      baseModelId: name,
    }));

    // Load user indexes from IndexedDB
    const userIndexes = await userIndexService.getUserIndexMetadata();
    const userModels = userIndexes.map((index) => ({
      title: `👤 ${index.name}`,
      value: index.id,
      isUserIndex: true,
      baseModelId: index.baseModelId,
      cellCount: index.cellCount,
      category: index.category,
    }));

    // Combine both lists
    availableModels.value = [...referenceModels, ...userModels];

    // Set default to 'scimilarity' if available and no model is selected
    if (
      !selectedModel.value ||
      !availableModels.value.find((m) => m.value === selectedModel.value)
    ) {
      if (modelNames.includes('scimilarity')) {
        selectedModel.value = 'scimilarity';
      } else if (modelNames.length > 0) {
        selectedModel.value = modelNames[0];
      }
    }
  } catch (error) {
    console.error('Error loading models:', error);
  }
};

const detectWebGPU = async () => {
  // Check max texture size as a proxy for GPU capability as apple silicon
  // has a limit of 16384 which is too low for our needs when embedding
  const gl = document.createElement('canvas').getContext('webgl');
  maxTextureSize.value = gl?.getParameter(gl.MAX_TEXTURE_SIZE);
  console.info('Max Texture Size:', maxTextureSize.value);

  try {
    const webGPUSupported = await isWebGPUSupported();
    hasWebGPU.value = webGPUSupported;
    useWebGPU.value = webGPUSupported;
  } catch (error) {
    console.error('WebGPU detection failed:', error);
    hasWebGPU.value = false;
    useWebGPU.value = false;
  }
};

// Function to check WebGPU support
async function isWebGPUSupported(): Promise<boolean> {
  try {
    if (!navigator.gpu) {
      console.log('WebGPU is not supported in this browser.');
      return false;
    }

    const adapter: GPUAdapter | null = await navigator.gpu.requestAdapter();
    if (!adapter) {
      console.log('No WebGPU adapter available.');
      return false;
    }

    console.log('WebGPU is supported and ready for ONNX Runtime.');
    return true;
  } catch (error) {
    console.error('Error checking WebGPU support:', error);
    return false;
  }
}

async function fetchSampleFile() {
  try {
    const sampleFileName = 'sample.h5ad';
    const response = await fetch(sampleFileName);
    const blob = await response.blob();
    const file = new File([blob], sampleFileName, { type: blob.type });
    selectedFile.value = file;
    console.log('Sample File:', file);
  } catch (error) {
    console.error('Error:', error);
  }
}

const loadCategoriesFromMetadata = async () => {
  try {
    const modelID = selectedModel.value;

    // Check if this is a user index
    const selectedModelInfo = availableModels.value.find((m) => m.value === modelID);
    if (selectedModelInfo?.isUserIndex) {
      // User indexes only have one category - the one selected at creation
      availableCategories.value = [selectedModelInfo.category || ''];
      selectedCategory.value = selectedModelInfo.category || '';
    } else {
      if (!selectedModelInfo) {
        availableCategories.value = [];
        selectedCategory.value = '';
        return;
      }
      const baseModelId = selectedModelInfo.baseModelId;
      const metadataResponse = await fetch(
        `${sitePath}/models/${baseModelId}/pumap/metadata.json`
      );
      const metadata = await metadataResponse.json();

      if (metadata.categories && typeof metadata.categories === 'object') {
        const categories = Object.keys(metadata.categories);
        availableCategories.value = categories;

        // Set default category
        if (!selectedCategory.value || !categories.includes(selectedCategory.value)) {
          if (baseModelId === 'scimilarity' && categories.includes('prediction')) {
            selectedCategory.value = 'prediction';
          } else if (baseModelId === 'brain' && categories.includes('CellType')) {
            selectedCategory.value = 'CellType';
          } else if (categories.length > 0) {
            selectedCategory.value = categories[0];
          }
        }
      }
      return;
    }
  } catch (error) {
    console.error('Error loading categories from metadata:', error);
    availableCategories.value = [];
  }
};

const loadUserIndexData = async (userIndexId: string) => {
  // Load the user index from IndexedDB
  const userIndex = await userIndexService.getUserIndex(userIndexId);
  if (!userIndex) {
    throw new Error('User index not found');
  }

  userRef.x.value = userIndex.x;
  userRef.y.value = userIndex.y;
  userRef.labelIndices.value = userIndex.labelIndices;

  // Force scatter plot update
  setTimeout(() => {
    scatterPlotRef.value?.forceUpdate();
  }, 100);
};

const loadReferenceData = async () => {
  isLoadingData.value = true;
  try {
    const selectedModelInfo = availableModels.value.find(
      (m) => m.value === selectedModel.value
    );

    if (!selectedModelInfo) {
      throw new Error('Selected model not found');
    }

    // Load the base reference model first

    // Load metadata to get categories information
    const metadataResponse = await fetch(
      `${sitePath}/models/${selectedModelInfo?.baseModelId}/pumap/metadata.json`
    );
    const metadata = await metadataResponse.json();

    // Get category labels from metadata
    baseRef.labels.value = metadata.categories?.[selectedCategory.value] as string[];

    // Store normalization parameters for query data
    xCenter.value = metadata.xCenter || 0;
    yCenter.value = metadata.yCenter || 0;
    maxRange.value = metadata.maxRange || 1;

    // Load binary files in parallel
    const [xResponse, yResponse, labelIndicesResponse] = await Promise.all([
      fetch(`${sitePath}/models/${selectedModelInfo?.baseModelId}/pumap/x.bin`),
      fetch(`${sitePath}/models/${selectedModelInfo?.baseModelId}/pumap/y.bin`),
      fetch(
        `${sitePath}/models/${selectedModelInfo?.baseModelId}/pumap/${selectedCategory.value}.bin`
      ),
    ]);

    // Check if all responses are successful
    if (!xResponse.ok || !yResponse.ok || !labelIndicesResponse.ok) {
      throw new Error(
        `Failed to fetch model data: ${xResponse.status}, ${yResponse.status}, ${labelIndicesResponse.status}`
      );
    }

    // Get array buffers from the responses
    const [xBuffer, yBuffer, labelIndicesBuffer] = await Promise.all([
      xResponse.arrayBuffer(),
      yResponse.arrayBuffer(),
      labelIndicesResponse.arrayBuffer(),
    ]);

    // Check if buffer sizes appear valid
    if (
      xBuffer.byteLength % 4 !== 0 ||
      yBuffer.byteLength % 4 !== 0 ||
      labelIndicesBuffer.byteLength % 2 !== 0
    ) {
      throw new Error(
        `Invalid buffer sizes: x=${xBuffer.byteLength}, y=${yBuffer.byteLength}, labelIndices=${labelIndicesBuffer.byteLength}`
      );
    }

    // Create typed arrays from the buffers that will NOT be sent to workers
    baseRef.x.value = new Float32Array(xBuffer);
    baseRef.y.value = new Float32Array(yBuffer);

    // Create ArrayBuffer based view for label indices as source
    const sourceLabelIndices = new Int16Array(labelIndicesBuffer);

    // Create a shared typed array for label indices to transfer to worker
    if (typeof SharedArrayBuffer === 'undefined') {
      throw new Error(
        'SharedArrayBuffer is not available. Ensure cross-origin isolation.'
      );
    }
    const sharedLabelIndicesBuffer = new SharedArrayBuffer(
      labelIndicesBuffer.byteLength
    );
    baseRef.labelIndices.value = new Int16Array(sharedLabelIndicesBuffer);
    baseRef.labelIndices.value.set(sourceLabelIndices);

    // Create a shared view of only the plotted cells for scatter plot
    // so that we can append to these the query cells
    baseRef.labelIndicesSubset.value = new Int16Array(
      baseRef.labelIndices.value.slice(0, baseRef.x.value.length)
    );

    console.log(
      `Loaded reference data: ${baseRef.x.value.length} points, ${baseRef.labels.value.length} label indices`
    );

    // Now load the user index data if applicable
    if (selectedModelInfo?.isUserIndex) {
      // Load user index data instead
      await loadUserIndexData(selectedModel.value);
    }
  } catch (error) {
    console.error('Error loading reference data:', error);
  } finally {
    isLoadingData.value = false;
  }
};

const start = async () => {
  console.log('Starting processing...', selectedFile.value?.name);

  // Clear any existing query data and state
  query.x.value = new Float32Array();
  query.y.value = new Float32Array();
  query.labelIndices.value = new Int16Array();
  labelCounts.value = {};
  totalLabeled.value = 0;
  totalNumCells.value = 0;
  totalProcessed.value = 0;
  canCreateIndex.value = false;
  currentUserIndexArtifacts.value = null;
  cellPositions.clear();
  cellIdToIndex.clear();

  // Start time tracking
  startTime.value = Date.now();

  // Set progress and running state
  progress.value = 0;
  isRunning.value = true;

  // Start scatter plot timer updates
  scatterPlotRef.value?.startTimerUpdates();

  // Create worker and start processing
  terminateWorker();
  createWorker();

  // Start the worker
  console.log(`Starting worker with modelID: ${selectedModel.value}`);

  // Check if this is a user index
  const selectedModelInfo = availableModels.value.find(
    (m) => m.value === selectedModel.value
  );
  const useUserIndex = selectedModelInfo?.isUserIndex || false;
  const baseModelId = useUserIndex
    ? selectedModelInfo?.baseModelId
    : selectedModel.value;

  if (useUserIndex) {
    console.log(`Using user index: ${selectedModel.value} based on ${baseModelId}`);
  } else {
    console.log(`Using reference model: ${selectedModel.value}`);
  }

  if (worker) {
    worker.postMessage({
      type: 'start',
      modelID: baseModelId, // Always use the base model ID for loading models
      modelsURL: `${sitePath}/models`,
      useUserIndex: useUserIndex,
      userIndexId: useUserIndex ? selectedModel.value : null,
      h5File: selectedFile.value,
      useWebGPU: useWebGPU.value,
      maxTextureSize: maxTextureSize.value,
      labelIndices: baseRef.labelIndices.value,
    });
  }
};

const stop = () => {
  statusMessage.value = 'Stopping processing...';
  isRunning.value = false;
  progress.value = 0;

  // Stop scatter plot timer updates and do final refresh
  scatterPlotRef.value?.stopTimerUpdates();
  scatterPlotRef.value?.forceUpdate();

  // Calculate elapsed time when stopped
  if (startTime.value) {
    const endTime = Date.now();
    const totalElapsed = Math.round((endTime - startTime.value) / 1000);
    const minutes = Math.floor(totalElapsed / 60);
    const seconds = totalElapsed % 60;
    const timeStr = minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
    statusMessage.value = `Stopped - Labeled ${totalProcessed.value.toLocaleString()} cells in ${timeStr}`;
  }

  terminateWorker();
};

const exportResultsToCSV = async () => {
  if (!query.labelIndices.value || query.labelIndices.value.length === 0) {
    console.error('No results to export');
    return;
  }

  try {
    // Generate CSV content
    let csv = 'cell_id,category_label,confidence\n';

    for (let i = 0; i < query.cellNames.value.length; i++) {
      const cellId = query.cellNames.value[i];
      const labelId = query.labelIndices.value[i];
      const confidence =
        query.confidences.value && query.confidences.value.length > i
          ? query.confidences.value[i]
          : undefined;
      const label =
        labelId >= 0 && labelId < baseRef.labels.value.length
          ? baseRef.labels.value[labelId]
          : 'Unknown';
      const confidenceStr = confidence !== undefined ? confidence.toFixed(3) : '1.000';
      csv += `"${cellId}","${label}",${confidenceStr}\n`;
    }

    // Create blob and download
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    const baseFilename = selectedFile.value?.name.replace('.h5ad', '') || 'results';
    const timestamp = new Date().toISOString().slice(0, 19).replace(/[:T-]/g, '_');
    link.download = `${baseFilename}_labels_${timestamp}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    console.log(`Exported ${query.cellNames.value.length} results to CSV`);
  } catch (error) {
    console.error('Failed to export results:', error);
  }
};

const handleDrawerToggle = () => {
  sidebarOpen.value = !sidebarOpen.value;
};

const confirmDeleteUserIndex = (indexId: string, indexName: string) => {
  indexToDelete.value = { id: indexId, name: indexName };
  deleteIndexModalOpen.value = true;
};

const deleteUserIndex = async () => {
  if (!indexToDelete.value) return;

  try {
    await userIndexService.deleteUserIndex(indexToDelete.value.id);
    await loadAvailableModels();

    // If the deleted index was selected, reset to default
    if (selectedModel.value === indexToDelete.value.id) {
      selectedModel.value =
        availableModels.value.find((m) => !m.isUserIndex)?.value || '';
    }

    statusMessage.value = 'User index deleted successfully';
    deleteIndexModalOpen.value = false;
    indexToDelete.value = null;
  } catch (error) {
    console.error('Error deleting user index:', error);
    errorMessage.value = `Failed to delete user index: ${error}`;
    errorModalOpen.value = true;
    deleteIndexModalOpen.value = false;
  }
};

const saveUserIndex = async () => {
  if (!currentUserIndexArtifacts.value || !newIndexName.value.trim()) {
    return;
  }

  try {
    // Transform artifacts to user index format with normalization
    const userIndex = userIndexService.transformArtifactsToUserIndex(
      newIndexName.value.trim(),
      selectedModel.value,
      currentUserIndexArtifacts.value,
      query.x.value,
      query.y.value,
      selectedCategory.value,
      baseRef.labels.value,
      query.labelIndices.value
    );

    userIndex.x = new Float32Array(query.x.value);
    userIndex.y = new Float32Array(query.y.value);

    // Save to IndexedDB
    await userIndexService.saveUserIndex(userIndex);

    // Clear the form and close modal
    newIndexName.value = '';
    createIndexModalOpen.value = false;
    canCreateIndex.value = false;
    currentUserIndexArtifacts.value = null;

    // Reload models list to include the new user index
    await loadAvailableModels();

    // Show success message
    statusMessage.value = `User index "${userIndex.name}" created successfully`;
  } catch (error) {
    console.error('Error saving user index:', error);
    errorMessage.value = `Failed to save user index: ${error}`;
    errorModalOpen.value = true;
  }
};

const handleImportIndex = async () => {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.json';

  input.onchange = async (event) => {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (!file) return;

    try {
      const indexId = await userIndexService.importUserIndex(file);

      // Reload models list to include the imported user index
      await loadAvailableModels();

      // Select the imported index
      selectedModel.value = indexId;

      statusMessage.value = 'User index imported successfully';
    } catch (error) {
      console.error('Error importing user index:', error);
      errorMessage.value = `Failed to import user index: ${error}`;
      errorModalOpen.value = true;
    }
  };

  input.click();
};

const handleRunStopClick = () => {
  if (isRunning.value) {
    stop();
  } else {
    start();
  }
};

const handleShareExportForSelected = () => {
  if (!selectedModelInfo.value?.isUserIndex) return;
  shareIndexInfo.value = {
    id: selectedModelInfo.value.value,
    name: selectedModelInfo.value.title,
  };
  shareModalOpen.value = true;
};

const handleShareExport = async () => {
  if (!shareIndexInfo.value) return;

  shareModalOpen.value = false;

  try {
    const blob = await userIndexService.exportUserIndex(shareIndexInfo.value.id);

    // Create download link
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${shareIndexInfo.value.name.replace('👤 ', '')}.json`;
    a.click();

    URL.revokeObjectURL(url);

    statusMessage.value = 'User reference exported successfully';
  } catch (error) {
    console.error('Error exporting user index:', error);
    errorMessage.value = `Failed to export user reference: ${error}`;
    errorModalOpen.value = true;
  } finally {
    shareIndexInfo.value = null;
  }
};

const handleFileSelect = (files: File | File[]) => {
  if (Array.isArray(files) && files.length > 0) {
    selectedFile.value = files[0];
  } else if (files && !Array.isArray(files)) {
    selectedFile.value = files;
  } else {
    selectedFile.value = null;
  }
};

const openFileDialog = () => {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.h5ad';

  input.onchange = (event) => {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  input.click();
};

const openGithub = () => {
  window.open('https://github.com/braingeneers/cytoverse', '_blank');
};

const updateIsMobile = () => {
  isMobile.value = window.innerWidth < 768;
};

// Watch for file changes
watch(selectedFile, (newFile) => {
  if (newFile) {
    fileList.value = [newFile];
  } else {
    fileList.value = [];
  }
});

// Initialize worker when model changes
watch(selectedModel, () => {
  terminateWorker();
  // Note: We don't create worker here, it is created on start
});

// Load categories when model changes
watch(selectedModel, async () => {
  // Clear existing query data and artifacts when model changes
  query.x.value = new Float32Array();
  query.y.value = new Float32Array();
  query.labelIndices.value = new Int16Array();
  labelCounts.value = {};
  totalLabeled.value = 0;
  cellPositions.clear();
  cellIdToIndex.clear();

  // Clear training data to refresh scatter plot
  baseRef.x.value = new Float32Array();
  baseRef.y.value = new Float32Array();
  baseRef.labelIndices.value = new Int16Array();
  baseRef.labels.value = [];

  // Clear user reference when the selected model changes
  userRef.x.value = new Float32Array();
  userRef.y.value = new Float32Array();
  userRef.labelIndices.value = new Int16Array();

  // Load categories for the new model
  await loadCategoriesFromMetadata();

  // Load training data for the new model/category
  if (selectedCategory.value) {
    await loadReferenceData();
  }
});

// Reload training data when selectedCategory changes
watch(selectedCategory, () => {
  if (selectedCategory.value) {
    loadReferenceData();
  }
});

onMounted(async () => {
  console.log('App mounted');

  // Initialize user index service
  await userIndexService.init();

  fetchSampleFile();
  detectWebGPU();
  await loadAvailableModels();
  await loadCategoriesFromMetadata();

  window.addEventListener('resize', updateIsMobile);
});

// Cleanup worker on unmount
onUnmounted(() => {
  terminateWorker();
  window.removeEventListener('resize', updateIsMobile);
});
</script>

<style scoped>
.app-container {
  height: 100vh;
  display: flex;
  position: relative;
}

.mini-sidebar {
  position: fixed;
  left: 0;
  top: 0;
  bottom: 0;
  width: 64px;
  background-color: #1e1e1e;
  border-right: 1px solid #424242;
  z-index: 10;
  display: flex;
  flex-direction: column;
}

.mini-toolbar {
  padding: 8px;
  border-bottom: 1px solid #424242;
}

.mini-actions {
  padding: 8px;
  flex: 1;
}

.main-drawer {
  background-color: #1e1e1e !important;
}

.drawer-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px;
  /* border-bottom: 1px solid #424242; */
}

.drawer-header h2 {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 500;
}

.header-actions {
  display: flex;
  gap: 4px;
  align-items: center;
}

.drawer-content {
  padding: 12px;
  overflow-y: auto;
  height: calc(100% - 72px);
}

.form-section {
  margin-bottom: 2px;
}

.stats-card {
  margin-bottom: 6px;
}

.stat-item {
  margin-bottom: 4px;
  font-size: 14px;
}

.stat-item:last-child {
  margin-bottom: 0;
}

.loading-text {
  color: #757575;
  font-style: italic;
}

.progress-section {
  margin-top: 8px;
  padding: 8px;
  background-color: #2a2a2a;
  border-radius: 4px;
}

.status-text {
  margin-bottom: 4px;
  font-size: 14px;
}

.progress-text {
  margin-top: 2px;
  text-align: right;
  font-size: 12px;
  color: #757575;
}

.labels-section {
  margin-top: 4px;
  padding-top: 2px;
}

.labels-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2px;
}

.labels-header h3 {
  margin: 0;
  font-size: 1.1rem;
}

.total-count {
  display: flex;
  justify-content: space-between;
  padding: 1px 0;
  font-weight: 600;
  border-bottom: 1px solid #424242;
  margin-bottom: 4px;
}

.label-count {
  display: flex;
  justify-content: space-between;
  padding: 2px 0;
  font-size: 14px;
}

.label-name {
  color: #e0e0e0;
  display: flex;
  align-items: center;
  gap: 8px;
}

.color-dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
}

.main-content {
  flex: 1;
  position: relative;
  height: 100vh;
  transition: margin-left 0.3s;
}

.loading-container {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  font-size: 18px;
  color: #757575;
}

.modal-description {
  margin-top: 16px;
  color: #757575;
  font-size: 14px;
}

.help-content p {
  margin-bottom: 16px;
}

.help-content ol {
  margin-left: 20px;
}

@media (max-width: 768px) {
  .main-content {
    margin-left: 0 !important;
  }
}
</style>

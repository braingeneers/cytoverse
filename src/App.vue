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
            <v-btn
              variant="text"
              icon
              size="small"
              @click="helpModalOpen = true"
            >
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
            <v-file-input
              :model-value="selectedFile ? [selectedFile] : []"
              accept=".h5ad"
              label="Select an .h5ad file"
              placeholder="Select an .h5ad file"
              prepend-icon=""
              variant="outlined"
              density="comfortable"
              @update:model-value="handleFileSelect"
            />
          </div>

          <!-- Model Selection -->
          <div class="form-section">
            <v-select
              data-testid="model-select-dropdown"
              v-model="selectedModel"
              :items="availableModels"
              label="Reference"
              :disabled="isRunning"
              variant="outlined"
              density="comfortable"
            >
              <template #item="{ item, props }">
                <v-list-item v-bind="props">
                  <template v-if="item.raw.isUserIndex" #append>
                    <v-btn
                      icon
                      size="x-small"
                      variant="text"
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
          </div>

          <!-- Category Selection -->
          <div class="form-section">
            <v-select
              data-testid="category-select-dropdown"
              v-model="selectedCategory"
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
                v-if="
                  xReferenceData &&
                  yReferenceData &&
                  categoryLabels &&
                  categoryData
                "
              >
                <div class="stat-item">
                  <strong>Reference Cells:</strong>
                  {{ categoryData.length.toLocaleString() }}
                </div>
                <div class="stat-item">
                  <strong>Plotted Cells:</strong>
                  {{ xReferenceData.length.toLocaleString() }}
                </div>
                <div class="stat-item">
                  <strong>Labels:</strong> {{ categoryLabels.length }}
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
              <v-radio
                :value="false"
                label="CPU"
                data-testid="radio-cpu-option"
              />
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
            block
            class="mt-2"
            @click="createIndexModalOpen = true"
          >
            Create Index
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
          <div
            v-if="Object.keys(labelCounts).length > 0"
            class="labels-section"
          >
            <div class="labels-header">
              <h3>{{ selectedCategory }}</h3>
              <div class="header-actions">
                <v-btn
                  variant="text"
                  icon
                  size="small"
                  :disabled="Object.keys(labelCounts).length === 0 || isRunning"
                  data-testid="download-button"
                  @click="exportResultsToCSV"
                  title="Download results as CSV"
                >
                  <v-icon>mdi-download</v-icon>
                </v-btn>
                <v-btn
                  variant="text"
                  icon
                  size="small"
                  :disabled="Object.keys(labelCounts).length === 0 || isRunning"
                  @click="shareModalOpen = true"
                >
                  <v-icon>mdi-share</v-icon>
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
              <span class="label-name">{{ label }}</span>
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
            ref="scatterPlotRef"
            v-else-if="
              xReferenceData &&
              yReferenceData &&
              categoryData &&
              categoryLabels.length > 0
            "
            :x-train-data="xReferenceData"
            :y-train-data="yReferenceData"
            :category-data="categoryData"
            :category-labels="categoryLabels"
            :x-test-data="xTestData"
            :y-test-data="yTestData"
            :test-data-labels="testDataLabels"
          />
          <div v-else class="loading-container">
            <div>No data available</div>
          </div>
        </div>
      </v-main>

      <!-- Share Modal -->
      <v-dialog v-model="shareModalOpen" max-width="500px">
        <v-card>
          <v-card-title>Share Embeddings</v-card-title>
          <v-card-text>
            <v-form>
              <v-text-field
                v-model="shareEmail"
                type="email"
                label="Email Address"
                placeholder="Enter your email"
                variant="outlined"
                density="comfortable"
              />
              <div class="modal-description">
                Get introduced to others with similar embeddings? Enter your
                email and we'll connect you. We'll never upload or share your
                raw data, only your embeddings. (Coming soon...)
              </div>
            </v-form>
          </v-card-text>
          <v-card-actions>
            <v-spacer />
            <v-btn @click="shareModalOpen = false">Cancel</v-btn>
            <v-btn
              color="primary"
              :disabled="!shareEmail.trim()"
              @click="handleShareLabels"
            >
              Share
            </v-btn>
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
                CytoVerse is a browser-based platform for single-cell RNA-seq
                analysis, designed for cell annotation using foundation model
                embeddings. It runs entirely in the browser, streaming h5ad
                files from local storage without uploading data or requiring
                server computation. It uses SCimilarity for cell embeddings and
                parametric UMAP for 2D visualization, leveraging large training
                datasets for accurate cell annotation. An Inverted File with
                Product Quantization (IVFPQ) enables fast approximate nearest
                neighbor searches across over 20 million samples. Built on
                WebAssembly and ONNX for high-speed processing, it supports
                unlimited streaming analysis of h5ad files via h5wasm. This
                enables distributed collaborative discovery, allowing
                researchers to explore shared embedding spaces to identify
                overlapping or complementary assays, particularly for
                perturbseq-driven research. The architecture ensures privacy,
                scalability, and collaborative potential without server
                dependency.
              </p>
              <p><strong>Getting Started:</strong></p>
              <ol>
                <li>Select an H5AD file containing single-cell RNA-seq data</li>
                <li>Choose a pre-trained model (Brain or Scimilarity)</li>
                <li>Select a category for cell type prediction</li>
                <li>Choose CPU or GPU processing (if available)</li>
                <li>Click Start to begin analysis</li>
              </ol>
              <p>
                The tool will display your cells in real-time as they are
                processed and labeled.
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

      <!-- Create Index Modal -->
      <v-dialog v-model="createIndexModalOpen" max-width="500px">
        <v-card>
          <v-card-title>Create User Index</v-card-title>
          <v-card-text>
            <p>
              Save the current labeled cells as a user index for future
              reference labeling.
            </p>
            <v-text-field
              v-model="newIndexName"
              label="Index Name"
              placeholder="Enter a name for your index"
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
import { openDB, deleteDB, DBSchema, IDBPDatabase } from 'idb';

import Worker from './worker?worker';
import ScatterPlotWebGL from './ScatterPlotWebGL.vue';
import { userIndexService, type UserIndex } from './userIndexService';

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

// IndexedDB schema
interface ResultsDB extends DBSchema {
  labels: {
    key: number;
    value: string;
  };
  results: {
    key: string;
    value: {
      labelId: number;
      x: number;
      y: number;
      confidence: number;
    };
  };
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
  }[]
>([]);
const isRunning = ref(false);
const hasWebGPU = ref(false);
const useWebGPU = ref(false);
const isMobile = ref(window.innerWidth < 768);

// Scatter plot data state - reference data (static)
const xReferenceData = ref<Float32Array | null>(null);
const yReferenceData = ref<Float32Array | null>(null);
const categoryData = ref<Int16Array | null>(null);
const categoryLabels = ref<string[]>([]);
const isLoadingData = ref(false);
const selectedCategory = ref('');
const availableCategories = ref<string[]>([]);

// Test data state - pre-allocated arrays for correct positioning
const xTestData = ref<number[]>([]);
const yTestData = ref<number[]>([]);
const testDataLabels = ref<number[]>([]);

// Processing state
const totalNumCells = ref(0);
const totalProcessed = ref(0);
const startTime = ref<number | null>(null);

// Labeling feedback state
const labelCounts = ref<{ [label: string]: number }>({});
const totalLabeled = ref(0);

// Modal states
const shareModalOpen = ref(false);
const shareEmail = ref('');
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
  partitionIds: Int32Array;
  pqCodes: Uint8Array[];
  labelIndices: Int32Array;
  x: Float32Array;
  y: Float32Array;
} | null>(null);

// Normalization parameters from metadata
const xCenter = ref(0);
const yCenter = ref(0);
const maxRange = ref(1);

// Worker management - single unified worker
let unifiedWorker: Worker | null = null;
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

let db: IDBPDatabase<ResultsDB> | null = null;

// Computed properties
const sortedLabelCounts = computed(() => {
  return Object.entries(labelCounts.value).sort(([, a], [, b]) => b - a);
});

// Helper function to normalize coordinates
const normalizeCoordinates = (x: number, y: number) => {
  const normalizedX = (x - xCenter.value) / (maxRange.value / 2);
  const normalizedY = (y - yCenter.value) / (maxRange.value / 2);
  return [normalizedX, normalizedY];
};

// Track cell ID to index mapping
const cellIdToIndex = new Map<string, number>();

// Handle cell updates from unified worker
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
      cellIndex = xTestData.value.length + newXData.length;
      cellIdToIndex.set(update.cellId, cellIndex);

      newXData.push(normalizedX);
      newYData.push(normalizedY);
      newLabels.push(update.labelId ?? -1);
    } else {
      // Update existing cell
      updatedIndices.push(cellIndex);
      xTestData.value[cellIndex] = normalizedX;
      yTestData.value[cellIndex] = normalizedY;
      if (update.labelId !== undefined) {
        testDataLabels.value[cellIndex] = update.labelId;
      }
    }

    // Update label counts if labeled
    if (
      update.labelId !== undefined &&
      update.labelId >= 0 &&
      update.labelId < categoryLabels.value.length
    ) {
      const label = categoryLabels.value[update.labelId];
      labelCounts.value[label] = (labelCounts.value[label] || 0) + 1;
      totalLabeled.value += 1;

      // Store in IndexedDB if available
      if (db) {
        const tx = db.transaction('results', 'readwrite');
        tx.store
          .put(
            {
              labelId: update.labelId,
              x: normalizedX,
              y: normalizedY,
              confidence: update.confidence ?? 0,
            },
            update.cellId
          )
          .catch((error) => {
            console.error('Failed to store test result:', error);
          });
      }
    }
  }

  // Batch append new data to arrays
  if (newXData.length > 0) {
    xTestData.value.push(...newXData);
    yTestData.value.push(...newYData);
    testDataLabels.value.push(...newLabels);
  }
};

// Terminate worker
const terminateWorker = () => {
  if (unifiedWorker) {
    console.log('Terminating unified worker...');
    unifiedWorker.terminate();
    unifiedWorker = null;
  }
};

// Create unified worker
const createUnifiedWorker = () => {
  console.log('Creating unified worker...');

  const worker = new Worker();

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

        // Store user index artifacts if available and we're using a reference model
        if (evt.data.userIndexArtifacts) {
          const selectedModelInfo = availableModels.value.find(
            (m) => m.value === selectedModel.value
          );
          const isUsingReferenceModel = !selectedModelInfo?.isUserIndex;

          currentUserIndexArtifacts.value = evt.data.userIndexArtifacts;
          canCreateIndex.value = isUsingReferenceModel; // Only allow creating from reference models
        }

        if (startTime.value) {
          const endTime = Date.now();
          const totalElapsed = Math.round((endTime - startTime.value) / 1000);
          const minutes = Math.floor(totalElapsed / 60);
          const seconds = totalElapsed % 60;
          const timeStr =
            minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
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

  unifiedWorker = worker;
};

// WebGPU detection
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
    }));

    // Load user indexes from IndexedDB
    const userIndexes = await userIndexService.getUserIndexMetadata();
    const userModels = userIndexes.map((index) => ({
      title: `👤 ${index.name}`,
      value: index.id,
      isUserIndex: true,
      baseModelId: index.baseModelId,
      cellCount: index.cellCount,
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
    // Fallback to hardcoded models
    availableModels.value = [
      { title: 'Brain', value: 'brain', isUserIndex: false },
      { title: 'SCimilarity', value: 'scimilarity', isUserIndex: false },
    ];
  }
};

const detectWebGPU = async () => {
  try {
    const webGPUSupported = await isWebGPUSupported();
    hasWebGPU.value = webGPUSupported;
    useWebGPU.value = webGPUSupported;
  } catch (error) {
    console.error('WebGPU detection failed:', error);
    hasWebGPU.value = false;
    useWebGPU.value = false;
  }

  // REMIND: Remove once we have a stable WebGPU implementation
  console.warn(
    'WebGPU is currently disabled as Apple silicon limits input vectors to 16384'
  );
  hasWebGPU.value = false;
  useWebGPU.value = false;
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
    const selectedModelInfo = availableModels.value.find(
      (m) => m.value === modelID
    );
    if (selectedModelInfo?.isUserIndex) {
      // For user indexes, load categories from the base model
      const baseModelId = selectedModelInfo.baseModelId;
      const metadataResponse = await fetch(
        `${sitePath}/models/${baseModelId}/pumap/metadata.json`
      );
      const metadata = await metadataResponse.json();

      if (metadata.categories && typeof metadata.categories === 'object') {
        const categories = Object.keys(metadata.categories);
        availableCategories.value = categories;

        // Set default category
        if (
          !selectedCategory.value ||
          !categories.includes(selectedCategory.value)
        ) {
          if (
            baseModelId === 'scimilarity' &&
            categories.includes('prediction')
          ) {
            selectedCategory.value = 'prediction';
          } else if (
            baseModelId === 'brain' &&
            categories.includes('CellType')
          ) {
            selectedCategory.value = 'CellType';
          } else if (categories.length > 0) {
            selectedCategory.value = categories[0];
          }
        }
      }
      return;
    }

    // Original logic for reference models
    const metadataResponse = await fetch(
      `${sitePath}/models/${modelID}/pumap/metadata.json`
    );
    const metadata = await metadataResponse.json();

    if (metadata.categories && typeof metadata.categories === 'object') {
      const categories = Object.keys(metadata.categories);
      availableCategories.value = categories;

      if (
        !selectedCategory.value ||
        !categories.includes(selectedCategory.value)
      ) {
        // REMIND: Should switch to specifying in metadata.json
        if (modelID === 'scimilarity' && categories.includes('prediction')) {
          selectedCategory.value = 'prediction';
        } else if (modelID === 'brain' && categories.includes('CellType')) {
          selectedCategory.value = 'CellType';
        } else if (categories.length > 0) {
          selectedCategory.value = categories[0];
        }
      }
    } else {
      availableCategories.value = [];
    }
  } catch (error) {
    console.error('Error loading categories from metadata:', error);
    availableCategories.value = [];
  }
};

const loadUserIndexData = async (userIndexId: string, baseModelId: string) => {
  // Load the user index from IndexedDB
  const userIndex = await userIndexService.getUserIndex(userIndexId);
  if (!userIndex) {
    throw new Error('User index not found');
  }

  // Load metadata from base model for normalization parameters
  const metadataResponse = await fetch(
    `${sitePath}/models/${baseModelId}/pumap/metadata.json`
  );
  const metadata = await metadataResponse.json();

  // Store normalization parameters
  xCenter.value = metadata.xCenter || 0;
  yCenter.value = metadata.yCenter || 0;
  maxRange.value = metadata.maxRange || 1;

  // Use the normalized user index coordinates directly
  // They are already normalized to [0,1] range when saved
  xReferenceData.value = userIndex.coordinates.x;
  yReferenceData.value = userIndex.coordinates.y;
  categoryLabels.value = userIndex.labels;

  console.log('Normalization params from metadata:', {
    xCenter: xCenter.value,
    yCenter: yCenter.value,
    maxRange: maxRange.value,
  });
  console.log('All user index coordinates:', {
    x: Array.from(userIndex.coordinates.x),
    y: Array.from(userIndex.coordinates.y),
  });
  console.log('Coordinate ranges:', {
    x: [
      Math.min(...userIndex.coordinates.x),
      Math.max(...userIndex.coordinates.x),
    ],
    y: [
      Math.min(...userIndex.coordinates.y),
      Math.max(...userIndex.coordinates.y),
    ],
  });
  console.log('Expected coordinate space based on metadata:', {
    xRange: [xCenter.value - maxRange.value, xCenter.value + maxRange.value],
    yRange: [yCenter.value - maxRange.value, yCenter.value + maxRange.value],
  });

  // Check for any outlier coordinates
  const outliers: Array<{ i: number; x: number; y: number }> = [];
  for (let i = 0; i < userIndex.coordinates.x.length; i++) {
    const x = userIndex.coordinates.x[i];
    const y = userIndex.coordinates.y[i];
    const expectedMinX = xCenter.value - maxRange.value;
    const expectedMaxX = xCenter.value + maxRange.value;
    const expectedMinY = yCenter.value - maxRange.value;
    const expectedMaxY = yCenter.value + maxRange.value;

    if (
      x < expectedMinX - 10 ||
      x > expectedMaxX + 10 ||
      y < expectedMinY - 10 ||
      y > expectedMaxY + 10
    ) {
      outliers.push({ i, x, y });
    }
  }
  if (outliers.length > 0) {
    console.warn('Outlier coordinates found:', outliers);
  } else {
    console.log('All coordinates are within expected bounds');
  }

  // Create category data by extracting label indices from all partitions
  // The coordinates are stored in the same order as they were originally processed
  // So we need to reconstruct the category array in the same order
  const categoryIndices = new Int16Array(userIndex.cellCount);
  const cellToLabel = new Map<number, number>();

  // Build a map from original cell index to label index
  for (const partition of Object.values(userIndex.partitions)) {
    for (let i = 0; i < partition.cellIndices.length; i++) {
      cellToLabel.set(partition.cellIndices[i], partition.labelIndices[i]);
    }
  }

  console.log(`Cell to label mapping size: ${cellToLabel.size}`);

  // Fill the category array in the same order as coordinates were stored
  for (let i = 0; i < userIndex.cellCount; i++) {
    const labelIndex = cellToLabel.get(i);
    if (labelIndex !== undefined) {
      categoryIndices[i] = labelIndex;
    } else {
      console.warn(`No label found for cell index ${i}, using 0`);
      categoryIndices[i] = 0;
    }
  }

  console.log(
    `CategoryIndices sample: [${Array.from(categoryIndices.slice(0, 10)).join(
      ', '
    )}]`
  );
  console.log(
    `Unique labels in categoryIndices: ${new Set(categoryIndices).size}`
  );

  categoryData.value = categoryIndices;

  console.log(
    `Loaded user index visualization: ${userIndex.cellCount} points, ${userIndex.labels.length} labels`
  );
  console.log(
    `User index coordinates range: x=[${Math.min(
      ...userIndex.coordinates.x
    )}, ${Math.max(...userIndex.coordinates.x)}], y=[${Math.min(
      ...userIndex.coordinates.y
    )}, ${Math.max(...userIndex.coordinates.y)}]`
  );
  console.log('xReferenceData length:', xReferenceData.value?.length);
  console.log('yReferenceData length:', yReferenceData.value?.length);
  console.log('categoryData length:', categoryData.value?.length);

  // Force scatter plot update
  setTimeout(() => {
    scatterPlotRef.value?.forceUpdate();
  }, 100);
};

const loadReferenceData = async () => {
  if (!selectedCategory.value) {
    return;
  }

  isLoadingData.value = true;
  try {
    const modelID = selectedModel.value;

    // Check if this is a user index
    const selectedModelInfo = availableModels.value.find(
      (m) => m.value === modelID
    );
    if (selectedModelInfo?.isUserIndex) {
      // Load user index data instead
      await loadUserIndexData(modelID, selectedModelInfo.baseModelId!);
      isLoadingData.value = false;
      return;
    }

    // Load metadata to get categories information
    const metadataResponse = await fetch(
      `${sitePath}/models/${modelID}/pumap/metadata.json`
    );
    const metadata = await metadataResponse.json();

    // Get category labels from metadata
    const labels = metadata.categories?.[selectedCategory.value];
    if (!labels || !Array.isArray(labels)) {
      throw new Error(
        `Category '${selectedCategory.value}' not found in metadata or invalid format`
      );
    }

    // Store normalization parameters for test data
    xCenter.value = metadata.xCenter || 0;
    yCenter.value = metadata.yCenter || 0;
    maxRange.value = metadata.maxRange || 1;

    // Load binary files in parallel
    const [xResponse, yResponse, categoryResponse] = await Promise.all([
      fetch(`${sitePath}/models/${modelID}/pumap/x.bin`),
      fetch(`${sitePath}/models/${modelID}/pumap/y.bin`),
      fetch(
        `${sitePath}/models/${modelID}/pumap/${selectedCategory.value}.bin`
      ),
    ]);

    // Check if all responses are successful
    if (!xResponse.ok)
      throw new Error(`Failed to fetch x.bin: ${xResponse.status}`);
    if (!yResponse.ok)
      throw new Error(`Failed to fetch y.bin: ${yResponse.status}`);
    if (!categoryResponse.ok)
      throw new Error(
        `Failed to fetch ${selectedCategory.value}.bin: ${categoryResponse.status}`
      );

    const [xBuffer, yBuffer, categoryBuffer] = await Promise.all([
      xResponse.arrayBuffer(),
      yResponse.arrayBuffer(),
      categoryResponse.arrayBuffer(),
    ]);

    // Convert binary buffers to typed arrays
    console.log(
      `Buffer sizes: x=${xBuffer.byteLength}, y=${yBuffer.byteLength}, category=${categoryBuffer.byteLength}`
    );

    // Check if buffer sizes are valid
    if (xBuffer.byteLength % 4 !== 0) {
      throw new Error(
        `X buffer size ${xBuffer.byteLength} is not a multiple of 4`
      );
    }
    if (yBuffer.byteLength % 4 !== 0) {
      throw new Error(
        `Y buffer size ${yBuffer.byteLength} is not a multiple of 4`
      );
    }
    if (categoryBuffer.byteLength % 2 !== 0) {
      throw new Error(
        `Category buffer size ${categoryBuffer.byteLength} is not a multiple of 2`
      );
    }

    const xData = new Float32Array(xBuffer);
    const yData = new Float32Array(yBuffer);
    const categoryIndices = new Int16Array(categoryBuffer);

    xReferenceData.value = xData;
    yReferenceData.value = yData;
    categoryData.value = categoryIndices;
    categoryLabels.value = labels;

    console.log(
      `Loaded reference data: ${xData.length} points, ${labels.length} category labels`
    );
    // Calculate ranges without spread operator to avoid call stack issues
    let xMin = xData[0], xMax = xData[0]
    let yMin = yData[0], yMax = yData[0]
    for (let i = 0; i < Math.min(1000, xData.length); i++) {
      if (xData[i] < xMin) xMin = xData[i]
      if (xData[i] > xMax) xMax = xData[i]
      if (yData[i] < yMin) yMin = yData[i] 
      if (yData[i] > yMax) yMax = yData[i]
    }
    console.log('Reference data coordinate ranges:', {
      x: [xMin, xMax],
      y: [yMin, yMax]
    })
    // console.log('First 5 reference coordinates:', {
    //   x: Array.from(xData.slice(0, 5)),
    //   y: Array.from(yData.slice(0, 5))
    // })
  } catch (error) {
    console.error('Error loading reference data:', error);
    // Switch to an available category silently
    // errorMessage.value = `Failed to load reference data: ${error}`
    // errorModalOpen.value = true
  } finally {
    isLoadingData.value = false;
  }
};

const start = async () => {
  console.log('Starting processing...', selectedFile.value?.name);

  // Clear any existing test data and state
  xTestData.value = [];
  yTestData.value = [];
  testDataLabels.value = [];
  labelCounts.value = {};
  totalLabeled.value = 0;
  totalNumCells.value = 0;
  totalProcessed.value = 0;
  canCreateIndex.value = false;
  currentUserIndexArtifacts.value = null;
  cellPositions.clear();
  cellIdToIndex.clear();

  // Initialize or clear the IndexedDB
  try {
    // Delete existing database if it exists
    if (db) {
      db.close();
      db = null;
    }
    await deleteDB('cytoverse');

    // Create new database
    db = await openDB<ResultsDB>('cytoverse', 1, {
      upgrade(db) {
        // Create categoryLabels store
        if (!db.objectStoreNames.contains('labels')) {
          db.createObjectStore('labels');
        }

        // Create testResults store
        if (!db.objectStoreNames.contains('results')) {
          db.createObjectStore('results');
        }
      },
    });

    // Store category labels
    const categoryTx = db.transaction('labels', 'readwrite');
    for (let i = 0; i < categoryLabels.value.length; i++) {
      await categoryTx.store.put(categoryLabels.value[i], i);
    }
    await categoryTx.done;
  } catch (error) {
    console.error('Failed to initialize IndexedDB:', error);
  }

  // Start time tracking
  startTime.value = Date.now();

  // Set progress and running state
  progress.value = 0;
  isRunning.value = true;

  // Start scatter plot timer updates
  scatterPlotRef.value?.startTimerUpdates();

  // Create worker and start processing
  terminateWorker();
  createUnifiedWorker();

  // Start the worker
  console.log(`Starting unified worker with modelID: ${selectedModel.value}`);

  // Check if this is a user index
  const selectedModelInfo = availableModels.value.find(
    (m) => m.value === selectedModel.value
  );
  const isUserIndex = selectedModelInfo?.isUserIndex || false;
  const baseModelId = isUserIndex
    ? selectedModelInfo?.baseModelId
    : selectedModel.value;

  if (unifiedWorker) {
    unifiedWorker.postMessage({
      type: 'start',
      modelsURL: `${sitePath}/models`,
      modelID: baseModelId, // Always use the base model ID for loading models
      userIndexId: isUserIndex ? selectedModel.value : null, // Pass user index ID if applicable
      isUserIndex,
      h5File: selectedFile.value,
      useWebGPU: useWebGPU.value,
      categoryData: categoryData.value || new Int16Array(0),
      categoryDataLength: categoryData.value?.length || 0,
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
  if (!db) {
    console.error('No database connection');
    return;
  }

  try {
    // Fetch all data from IndexedDB
    const categoryTx = db.transaction('labels', 'readonly');
    const storedLabels = await categoryTx.store.getAll();
    const storedLabelKeys = await categoryTx.store.getAllKeys();
    await categoryTx.done;

    const resultsTx = db.transaction('results', 'readonly');
    const testResults = await resultsTx.store.getAll();
    const testResultKeys = await resultsTx.store.getAllKeys();
    await resultsTx.done;

    // Create label lookup map
    const labelMap: { [id: number]: string } = {};
    storedLabels.forEach((label, index) => {
      const labelId = storedLabelKeys[index];
      labelMap[labelId] = label;
    });

    // Generate CSV content
    let csv = 'cell_id,category_label,confidence\n';

    testResults.forEach((result, index) => {
      const vectorId = testResultKeys[index];
      const label =
        result.labelId >= 0 ? labelMap[result.labelId] || 'Unknown' : 'Unknown';
      const confidence =
        result.confidence !== undefined
          ? result.confidence.toFixed(3)
          : '1.000';
      csv += `"${vectorId}","${label}",${confidence}\n`;
    });

    // Create blob and download
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    const baseFilename =
      selectedFile.value?.name.replace('.h5ad', '') || 'results';
    const timestamp = new Date()
      .toISOString()
      .slice(0, 19)
      .replace(/[:T-]/g, '_');
    link.download = `${baseFilename}_labels_${timestamp}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    console.log(`Exported ${testResults.length} results to CSV`);
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
      categoryLabels.value,
      {
        xCenter: xCenter.value,
        yCenter: yCenter.value,
        maxRange: maxRange.value
      }
    );

    // Debug: Check the structure before saving
    console.log('User index structure:', {
      id: userIndex.id,
      name: userIndex.name,
      baseModelId: userIndex.baseModelId,
      cellCount: userIndex.cellCount,
      partitionCount: Object.keys(userIndex.partitions).length,
      labelsType: typeof userIndex.labels,
      labelsIsArray: Array.isArray(userIndex.labels),
      coordinatesXType: userIndex.coordinates.x?.constructor?.name,
      coordinatesYType: userIndex.coordinates.y?.constructor?.name,
    });

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

const handleRunStopClick = () => {
  if (isRunning.value) {
    stop();
  } else {
    start();
  }
};

const handleShareLabels = () => {
  console.log('Sharing labels to:', shareEmail.value);
  // TODO: Implement actual sharing logic
  shareModalOpen.value = false;
  shareEmail.value = '';
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
  // Clear existing test data and artifacts when model changes
  xTestData.value = [];
  yTestData.value = [];
  testDataLabels.value = [];
  labelCounts.value = {};
  totalLabeled.value = 0;
  cellPositions.clear();
  cellIdToIndex.clear();

  // Clear training data to refresh scatter plot
  xReferenceData.value = null;
  yReferenceData.value = null;
  categoryData.value = null;
  categoryLabels.value = [];

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
  loadAvailableModels();
  loadCategoriesFromMetadata();

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
  padding: 16px;
  border-bottom: 1px solid #424242;
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
  padding: 16px;
  overflow-y: auto;
  height: calc(100% - 72px);
}

.form-section {
  margin-bottom: 16px;
}

.stats-card {
  margin-bottom: 16px;
}

.stat-item {
  margin-bottom: 8px;
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
  margin-top: 16px;
  padding: 16px;
  background-color: #2a2a2a;
  border-radius: 4px;
}

.status-text {
  margin-bottom: 8px;
  font-size: 14px;
}

.progress-text {
  margin-top: 4px;
  text-align: right;
  font-size: 12px;
  color: #757575;
}

.labels-section {
  margin-top: 24px;
  padding-top: 16px;
  border-top: 1px solid #424242;
}

.labels-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.labels-header h3 {
  margin: 0;
  font-size: 1.1rem;
}

.total-count {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  font-weight: 600;
  border-bottom: 1px solid #424242;
  margin-bottom: 8px;
}

.label-count {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
  font-size: 14px;
}

.label-name {
  color: #e0e0e0;
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

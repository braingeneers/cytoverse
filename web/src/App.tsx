import { useEffect, useState, useCallback, useRef } from 'react'
import { ThemeProvider } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'
import { createTheme } from '@mui/material/styles'
import {
  Box,
  Drawer,
  Toolbar,
  Typography,
  IconButton,
  LinearProgress,
  useMediaQuery,
  Button,
  FormControl,
  RadioGroup,
  FormControlLabel,
  Radio,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  InputLabel,
  Tooltip,
} from '@mui/material'
import MenuIcon from '@mui/icons-material/Menu'
import GitHubIcon from '@mui/icons-material/GitHub'
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import StopIcon from '@mui/icons-material/Stop'
import ShareIcon from '@mui/icons-material/Share'
import DownloadIcon from '@mui/icons-material/Download'
import HelpOutlineIcon from '@mui/icons-material/HelpOutline'
import InfoIcon from '@mui/icons-material/Info'
import { MuiFileInput } from 'mui-file-input'
import { tableFromIPC, Vector } from 'apache-arrow'
import { openDB, deleteDB, DBSchema, IDBPDatabase } from 'idb'

import EmbeddingWorker from './embedder?worker'
import LabelerWorker from './labeler?worker'
import ScatterPlotWebGL from './ScatterPlotWebGL'

const drawerWidth = 320
const miniDrawerWidth = 64

// Create a dark theme
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
  },
})

// Types for embedding batches
interface EmbeddingBatch {
  test_vector_ids: string[]
  pq_embeddings: Uint8Array
  umap_coordinates: number[][]
  start_index: number
  end_index: number
}

// IndexedDB schema
interface ResultsDB extends DBSchema {
  labels: {
    key: number
    value: string
  }
  results: {
    key: string
    value: {
      labelId: number
      x: number
      y: number
      confidence: number
    }
  }
}

let db: IDBPDatabase<ResultsDB> | null = null

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [statusMessage, setStatusMessage] = useState('')
  const [progress, setProgress] = useState(0)
  const [selectedModel, setSelectedModel] = useState<string>('scimilarity')

  const [isRunning, setIsRunning] = useState(false)
  const [hasWebGPU, setHasWebGPU] = useState(false)
  const [useWebGPU, setUseWebGPU] = useState(false)
  const isMobile = useMediaQuery(theme.breakpoints.down('md'))

  // Scatter plot data state - training data (static)
  const [xTrainData, setXTrainData] = useState<Vector | null>(null)
  const [yTrainData, setYTrainData] = useState<Vector | null>(null)
  const [categoryData, setCategoryData] = useState<Vector | null>(null)
  const [categoryLabels, setCategoryLabels] = useState<string[]>([])
  const [isLoadingData, setIsLoadingData] = useState(false)
  const [selectedCategory, setSelectedCategory] = useState<string>('')
  const [availableCategories, setAvailableCategories] = useState<string[]>([])

  // Test data state - pre-allocated arrays for correct positioning
  const [xTestData, setXTestData] = useState<number[]>([])
  const [yTestData, setYTestData] = useState<number[]>([])
  const [testDataLabels, setTestDataLabels] = useState<number[]>([])

  // Processing state
  const totalNumCells = useRef(0)
  const totalProcessed = useRef(0)
  const startTime = useRef<number | null>(null)

  // Labeling feedback state
  const [labelCounts, setLabelCounts] = useState<{ [label: string]: number }>({})
  const [totalLabeled, setTotalLabeled] = useState<number>(0)

  // Modal states
  const [shareModalOpen, setShareModalOpen] = useState(false)
  const [shareEmail, setShareEmail] = useState('')
  const [helpModalOpen, setHelpModalOpen] = useState(false)
  const [errorModalOpen, setErrorModalOpen] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')

  // Normalization parameters from metadata
  const [xCenter, setXCenter] = useState<number>(0)
  const [yCenter, setYCenter] = useState<number>(0)
  const [maxRange, setMaxRange] = useState<number>(1)

  // Worker management - simplified approach
  const embedderWorker = useRef<Worker | null>(null)
  const labelerWorkers = useRef<Worker[]>([])
  const numLabelers = Math.max(1, Math.floor(navigator.hardwareConcurrency / 3))
  const labelerBusy = useRef<boolean[]>(new Array(numLabelers).fill(false))
  const pendingBatches = useRef<EmbeddingBatch[]>([])

  // Site path calculation
  const sitePath =
    window.location.origin +
    window.location.pathname.slice(0, window.location.pathname.lastIndexOf('/'))

  console.log(
    `Will create ${numLabelers} labeler workers (${navigator.hardwareConcurrency} cores available)`
  )

  // Helper function to normalize coordinates
  const normalizeCoordinates = useCallback(
    (x: number, y: number) => {
      const normalizedX = (x - xCenter) / (maxRange / 2)
      const normalizedY = (y - yCenter) / (maxRange / 2)
      return [normalizedX, normalizedY]
    },
    [xCenter, yCenter, maxRange]
  )

  // Assign pending batches to available labelers
  const assignBatchToLabeler = useCallback(() => {
    if (pendingBatches.current.length === 0 || labelerWorkers.current.length === 0) {
      console.log(
        `No batches to assign: pending=${pendingBatches.current.length}, workers=${labelerWorkers.current.length}`
      )
      return
    }

    console.log(
      `Assigning batches: ${
        pendingBatches.current.length
      } pending, labelers busy: ${labelerBusy.current.map((b, i) => `${i}:${b}`).join(', ')}`
    )

    // Find an available labeler
    for (let i = 0; i < numLabelers; i++) {
      if (!labelerBusy.current[i] && labelerWorkers.current[i]) {
        const batch = pendingBatches.current.shift()
        if (!batch || !batch.test_vector_ids || !batch.pq_embeddings) {
          console.error('Invalid batch data:', batch)
          continue
        }

        console.log(`Assigning batch [${batch.start_index}-${batch.end_index}] to labeler ${i}`)
        labelerWorkers.current[i].postMessage({
          type: 'embedding',
          test_vector_ids: batch.test_vector_ids,
          pq_embeddings: batch.pq_embeddings,
          umap_coordinates: batch.umap_coordinates,
          start_index: batch.start_index,
          end_index: batch.end_index,
        })
        labelerBusy.current[i] = true
        break
      }
    }
  }, [numLabelers])

  // Terminate all workers
  const terminateWorkers = useCallback(() => {
    if (embedderWorker.current) {
      console.log('Terminating embedder worker...')
      embedderWorker.current.terminate()
      embedderWorker.current = null
    }

    labelerWorkers.current.forEach((worker, idx) => {
      console.log(`Terminating labeler worker ${idx}...`)
      worker.terminate()
    })
    labelerWorkers.current = []
    labelerBusy.current = new Array(numLabelers).fill(false)
  }, [numLabelers])

  // Create labeler workers
  const createLabelerWorkers = useCallback(() => {
    console.log('Creating labeler workers...')

    const workers: Worker[] = []
    for (let i = 0; i < numLabelers; i++) {
      const labeler = new LabelerWorker()
      workers.push(labeler)

      // Initialize labeler
      console.log(`Initializing labeler ${i} with modelID: ${selectedModel}`)
      labeler.postMessage({
        type: 'start',
        modelsURL: `${sitePath}/models`,
        modelID: selectedModel,
        categoryData: categoryData?.data || new Int32Array(0),
        categoryDataLength: categoryData?.length || 0,
      })

      // Mark as busy until initialized
      labelerBusy.current[i] = true

      // Handle labeler messages
      labeler.onmessage = (evt) => {
        switch (evt.data.type) {
          case 'status':
            console.log(`Labeler ${i} status:`, evt.data.message)
            if (evt.data.message && evt.data.message.includes('initialized successfully')) {
              labelerBusy.current[i] = false
              console.log(`Labeler ${i} is ready`)
              // Try to assign any pending batches
              assignBatchToLabeler()
            }
            break
          case 'labeled':
            console.log('Received labeled batch:', evt.data.umap_coordinates?.length, 'points')
            console.log(`Labeled batch indices: ${evt.data.start_index} to ${evt.data.end_index}`)
            labelerBusy.current[i] = false

            // Process labeling results
            if (
              evt.data.train_vector_ids &&
              evt.data.label_ids &&
              evt.data.confidences &&
              categoryLabels.length > 0
            ) {
              const newLabelCounts: { [label: string]: number } = {}
              let validLabels = 0

              // First, calculate counts and prepare label updates
              const labelUpdates: { index: number; categoryIndex: number; confidence: number }[] =
                []

              for (let idx = 0; idx < evt.data.train_vector_ids.length; idx++) {
                const categoryIndex = evt.data.label_ids[idx]
                const confidence = evt.data.confidences[idx]

                if (categoryIndex >= 0 && categoryIndex < categoryLabels.length) {
                  const label = categoryLabels[categoryIndex]
                  newLabelCounts[label] = (newLabelCounts[label] || 0) + 1
                  validLabels++
                }

                labelUpdates.push({ index: evt.data.start_index + idx, categoryIndex, confidence })

                // Store in IndexedDB if available
                if (
                  db &&
                  evt.data.test_vector_ids &&
                  evt.data.umap_coordinates &&
                  evt.data.test_vector_ids[idx] &&
                  evt.data.umap_coordinates[idx]
                ) {
                  const tx = db.transaction('results', 'readwrite')
                  tx.store
                    .put(
                      {
                        labelId: categoryIndex,
                        x: evt.data.umap_coordinates[idx][0],
                        y: evt.data.umap_coordinates[idx][1],
                        confidence: confidence,
                      },
                      evt.data.test_vector_ids[idx]
                    )
                    .catch((error) => {
                      console.error('Failed to store test result:', error)
                    })
                }
              }

              // Batch update test labels
              setTestDataLabels((prev) => {
                const updated = [...prev]
                for (const { index, categoryIndex } of labelUpdates) {
                  if (index < updated.length) {
                    updated[index] = categoryIndex
                  }
                }
                return updated
              })

              // Update counts and progress
              setLabelCounts((prev) => {
                const updated = { ...prev }
                for (const [label, count] of Object.entries(newLabelCounts)) {
                  updated[label] = (updated[label] || 0) + count
                }
                return updated
              })

              setTotalLabeled((prevLabeled) => prevLabeled + validLabels)
              totalProcessed.current += evt.data.train_vector_ids.length

              const progressPercent = Math.min(
                100,
                Math.round((totalProcessed.current / totalNumCells.current) * 100)
              )
              setProgress(progressPercent)
              setStatusMessage(
                `Processed ${totalProcessed.current} of ${totalNumCells.current} cells...`
              )

              if (
                totalProcessed.current >= totalNumCells.current &&
                pendingBatches.current.length === 0
              ) {
                setIsRunning(false)
                if (startTime.current) {
                  const endTime = Date.now()
                  const totalElapsed = Math.round((endTime - startTime.current) / 1000)
                  const minutes = Math.floor(totalElapsed / 60)
                  const seconds = totalElapsed % 60
                  const timeStr = minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`
                  setStatusMessage(
                    `Complete - Labeled ${totalProcessed.current.toLocaleString()} cells in ${timeStr}`
                  )
                } else {
                  setStatusMessage('Processing complete')
                }
              }
            }

            // Try to assign next batch
            assignBatchToLabeler()
            break
          case 'error':
            console.error(`Labeler ${i} error:`, evt.data.error)
            labelerBusy.current[i] = false
            assignBatchToLabeler()
            break
        }
      }
    }

    labelerWorkers.current = workers
  }, [selectedModel, sitePath, numLabelers, assignBatchToLabeler, categoryData, categoryLabels])

  // Create embedder worker
  const createEmbedderWorker = useCallback(() => {
    console.log('Creating embedder worker...')

    const embedder = new EmbeddingWorker()

    embedder.onmessage = (evt) => {
      switch (evt.data.type) {
        case 'status':
          setStatusMessage(evt.data.message)
          break
        case 'modelDownloadProgress':
          setProgress(
            Math.min(100, Math.round((evt.data.countFinished / evt.data.totalToProcess) * 100))
          )
          setStatusMessage('Downloading model...')
          break
        case 'embedding': {
          console.log('Received embedding batch:', evt.data.umap_coordinates?.length, 'points')
          console.log(`Received embeddings: ${evt.data.start_index} to ${evt.data.end_index}`)
          totalNumCells.current = evt.data.totalToProcess

          // Initialize arrays if needed
          setXTestData((prev) => {
            if (prev.length < evt.data.totalToProcess) {
              return new Array(evt.data.totalToProcess).fill(0)
            }
            return prev
          })
          setYTestData((prev) => {
            if (prev.length < evt.data.totalToProcess) {
              return new Array(evt.data.totalToProcess).fill(0)
            }
            return prev
          })
          setTestDataLabels((prev) => {
            if (prev.length < evt.data.totalToProcess) {
              return new Array(evt.data.totalToProcess).fill(-1)
            }
            return prev
          })

          // Plot coordinates at correct indices
          if (evt.data.umap_coordinates && evt.data.umap_coordinates.length > 0) {
            setXTestData((prev) => {
              const updated = [...prev]
              for (let i = 0; i < evt.data.umap_coordinates.length; i++) {
                const coordinate = evt.data.umap_coordinates[i]
                if (coordinate && coordinate.length >= 2) {
                  const [normalizedX] = normalizeCoordinates(coordinate[0], coordinate[1])
                  updated[evt.data.start_index + i] = normalizedX
                }
              }
              return updated
            })
            setYTestData((prev) => {
              const updated = [...prev]
              for (let i = 0; i < evt.data.umap_coordinates.length; i++) {
                const coordinate = evt.data.umap_coordinates[i]
                if (coordinate && coordinate.length >= 2) {
                  const [, normalizedY] = normalizeCoordinates(coordinate[0], coordinate[1])
                  updated[evt.data.start_index + i] = normalizedY
                }
              }
              return updated
            })
          }

          // Add to pending queue and try to assign to labeler
          const batch: EmbeddingBatch = {
            test_vector_ids: evt.data.test_vector_ids,
            pq_embeddings: evt.data.pq_embeddings,
            umap_coordinates: evt.data.umap_coordinates,
            start_index: evt.data.start_index,
            end_index: evt.data.end_index,
          }
          pendingBatches.current.push(batch)
          assignBatchToLabeler()
          break
        }
        case 'finished':
          console.log('Embedder finished processing')
          console.log(`Pending batches: ${pendingBatches.current.length}`)
          // Try to assign any remaining pending batches
          for (let j = 0; j < numLabelers; j++) {
            if (pendingBatches.current.length > 0) {
              assignBatchToLabeler()
            }
          }
          break
        case 'error':
          console.error('Embedder error:', evt.data.error)
          setIsRunning(false)
          setStatusMessage('')
          setErrorMessage(evt.data.error.toString())
          setErrorModalOpen(true)
          break
      }
    }

    embedderWorker.current = embedder
  }, [assignBatchToLabeler, normalizeCoordinates])

  // WebGPU detection
  const detectWebGPU = useCallback(async () => {
    try {
      const webGPUSupported = await isWebGPUSupported()
      setHasWebGPU(webGPUSupported)
      setUseWebGPU(webGPUSupported)
    } catch (error) {
      console.error('WebGPU detection failed:', error)
      setHasWebGPU(false)
      setUseWebGPU(false)
    }

    // REMIND: Remove once we have a stable WebGPU implementation
    console.warn('WebGPU is currently disabled as Apple silicon limits input vectors to 16384')
    setHasWebGPU(false)
    setUseWebGPU(false)
  }, [])

  // Function to check WebGPU support
  async function isWebGPUSupported(): Promise<boolean> {
    try {
      if (!navigator.gpu) {
        console.log('WebGPU is not supported in this browser.')
        return false
      }

      const adapter: GPUAdapter | null = await navigator.gpu.requestAdapter()
      if (!adapter) {
        console.log('No WebGPU adapter available.')
        return false
      }

      console.log('WebGPU is supported and ready for ONNX Runtime.')
      return true
    } catch (error) {
      console.error('Error checking WebGPU support:', error)
      return false
    }
  }

  async function fetchSampleFile() {
    try {
      const sampleFileName = 'sample.h5ad'
      const response = await fetch(sampleFileName)
      const blob = await response.blob()
      const file = new File([blob], sampleFileName, { type: blob.type })
      setSelectedFile(file)
      console.log('Sample File:', file)
    } catch (error) {
      console.error('Error:', error)
    }
  }

  const loadCategoriesFromMetadata = useCallback(async () => {
    try {
      const modelID = selectedModel
      const metadataResponse = await fetch(`${sitePath}/models/${modelID}/pumap/metadata.json`)
      const metadata = await metadataResponse.json()

      if (metadata.categories && typeof metadata.categories === 'object') {
        const categories = Object.keys(metadata.categories)
        setAvailableCategories(categories)

        if (!selectedCategory || !categories.includes(selectedCategory)) {
          // REMIND: Should switch to specifying in metadata.json
          if (modelID === 'scimilarity' && categories.includes('prediction')) {
            setSelectedCategory('prediction')
          } else if (modelID === 'brain' && categories.includes('CellType')) {
            setSelectedCategory('CellType')
          } else if (categories.length > 0) {
            setSelectedCategory(categories[0])
          }
        }
      } else {
        setAvailableCategories([])
      }
    } catch (error) {
      console.error('Error loading categories from metadata:', error)
      setAvailableCategories([])
    }
  }, [selectedModel, sitePath, selectedCategory])

  const loadTrainingData = useCallback(async () => {
    if (!selectedCategory) {
      return
    }

    setIsLoadingData(true)
    try {
      const modelID = selectedModel

      // Load metadata to get categories information
      const metadataResponse = await fetch(`${sitePath}/models/${modelID}/pumap/metadata.json`)
      const metadata = await metadataResponse.json()

      // Get category labels from metadata
      const labels = metadata.categories?.[selectedCategory]
      if (!labels || !Array.isArray(labels)) {
        throw new Error(`Category '${selectedCategory}' not found in metadata or invalid format`)
      }

      // Store normalization parameters for test data
      setXCenter(metadata.xCenter || 0)
      setYCenter(metadata.yCenter || 0)
      setMaxRange(metadata.maxRange || 1)

      // Load Arrow files in parallel
      const [xResponse, yResponse, categoryResponse] = await Promise.all([
        fetch(`${sitePath}/models/${modelID}/pumap/x.arrow`),
        fetch(`${sitePath}/models/${modelID}/pumap/y.arrow`),
        fetch(`${sitePath}/models/${modelID}/pumap/${selectedCategory}.arrow`),
      ])

      const [xBuffer, yBuffer, categoryBuffer] = await Promise.all([
        xResponse.arrayBuffer(),
        yResponse.arrayBuffer(),
        categoryResponse.arrayBuffer(),
      ])

      // Convert Arrow buffers to tables
      const xTable = tableFromIPC(new Uint8Array(xBuffer))
      const yTable = tableFromIPC(new Uint8Array(yBuffer))
      const categoryTable = tableFromIPC(new Uint8Array(categoryBuffer))

      // Extract vectors
      const xVector = xTable.getChild('x')!
      const yVector = yTable.getChild('y')!
      const categoryVector = categoryTable.getChild(selectedCategory)!

      setXTrainData(xVector)
      setYTrainData(yVector)
      setCategoryData(categoryVector)
      setCategoryLabels(labels)

      console.log(
        `Loaded training data: ${xVector.length} points, ${labels.length} category labels`
      )
    } catch (error) {
      console.error('Error loading training data:', error)
      // Switch to an available category silently
      // setErrorMessage(`Failed to load training data: ${error}`)
      // setErrorModalOpen(true)
    } finally {
      setIsLoadingData(false)
    }
  }, [sitePath, selectedCategory, selectedModel])

  // Initialize workers when model changes
  useEffect(() => {
    terminateWorkers()
    // Note: We don't create workers here, they are created on start
  }, [selectedModel, terminateWorkers])

  useEffect(() => {
    console.log('App mounted')
    fetchSampleFile()
    detectWebGPU()
    loadCategoriesFromMetadata()
  }, [detectWebGPU, loadCategoriesFromMetadata])

  // Load categories when model changes
  useEffect(() => {
    loadCategoriesFromMetadata()
  }, [selectedModel, loadCategoriesFromMetadata])

  // Reload training data when selectedCategory changes
  useEffect(() => {
    if (selectedCategory) {
      loadTrainingData()
    }
  }, [selectedCategory, loadTrainingData])

  const start = async () => {
    console.log('Starting embedding...', selectedFile?.name)

    // Clear any existing test data and state
    setXTestData([])
    setYTestData([])
    setTestDataLabels([])
    setLabelCounts({})
    setTotalLabeled(0)
    totalNumCells.current = 0
    totalProcessed.current = 0
    pendingBatches.current = []

    // Initialize or clear the IndexedDB
    try {
      // Delete existing database if it exists
      if (db) {
        db.close()
        db = null
      }
      await deleteDB('cytoverse')

      // Create new database
      db = await openDB<ResultsDB>('cytoverse', 1, {
        upgrade(db) {
          // Create categoryLabels store
          if (!db.objectStoreNames.contains('labels')) {
            db.createObjectStore('labels')
          }

          // Create testResults store
          if (!db.objectStoreNames.contains('results')) {
            db.createObjectStore('results')
          }
        },
      })

      // Database is now fresh - no need to clear data

      // Store category labels
      const categoryTx = db.transaction('labels', 'readwrite')
      for (let i = 0; i < categoryLabels.length; i++) {
        await categoryTx.store.put(categoryLabels[i], i)
      }
      await categoryTx.done
    } catch (error) {
      console.error('Failed to initialize IndexedDB:', error)
    }

    // Start time tracking
    startTime.current = Date.now()

    // Set progress and running state
    setProgress(0)
    setIsRunning(true)

    // Create workers and start processing
    terminateWorkers()
    createLabelerWorkers()
    createEmbedderWorker()

    // Start the embedder
    console.log(`Starting embedder with modelID: ${selectedModel}`)
    if (embedderWorker.current) {
      embedderWorker.current.postMessage({
        type: 'start',
        modelsURL: `${sitePath}/models`,
        modelID: selectedModel,
        h5File: selectedFile,
        cellRangePercent: 100,
        useWebGPU: useWebGPU,
      })
    }
  }

  const stop = () => {
    setStatusMessage('Stopping processing...')
    setIsRunning(false)
    setProgress(0)

    // Calculate elapsed time when stopped
    if (startTime.current) {
      const endTime = Date.now()
      const totalElapsed = Math.round((endTime - startTime.current) / 1000)
      const minutes = Math.floor(totalElapsed / 60)
      const seconds = totalElapsed % 60
      const timeStr = minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`
      setStatusMessage(
        `Stopped - Labeled ${totalProcessed.current.toLocaleString()} cells in ${timeStr}`
      )
    }

    terminateWorkers()
  }

  const exportResultsToCSV = async () => {
    if (!db) {
      console.error('No database connection')
      return
    }

    try {
      // Fetch all data from IndexedDB
      const categoryTx = db.transaction('labels', 'readonly')
      const storedLabels = await categoryTx.store.getAll()
      const storedLabelKeys = await categoryTx.store.getAllKeys()
      await categoryTx.done

      const resultsTx = db.transaction('results', 'readonly')
      const testResults = await resultsTx.store.getAll()
      const testResultKeys = await resultsTx.store.getAllKeys()
      await resultsTx.done

      // Create label lookup map
      const labelMap: { [id: number]: string } = {}
      storedLabels.forEach((label, index) => {
        const labelId = storedLabelKeys[index]
        labelMap[labelId] = label
      })

      // Generate CSV content
      let csv = 'cell_id,category_label,confidence\n'

      testResults.forEach((result, index) => {
        const vectorId = testResultKeys[index]
        const label = result.labelId >= 0 ? labelMap[result.labelId] || 'Unknown' : 'Unknown'
        const confidence = result.confidence !== undefined ? result.confidence.toFixed(3) : '1.000'
        csv += `"${vectorId}","${label}",${confidence}\n`
      })

      // Create blob and download
      const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      const baseFilename = selectedFile?.name.replace('.h5ad', '') || 'results'
      const timestamp = new Date().toISOString().slice(0, 19).replace(/[:T-]/g, '_')
      link.download = `${baseFilename}_labels_${timestamp}.csv`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)

      console.log(`Exported ${testResults.length} results to CSV`)
    } catch (error) {
      console.error('Failed to export results:', error)
    }
  }

  const handleDrawerToggle = () => {
    setSidebarOpen(!sidebarOpen)
  }

  const handleRunStopClick = () => {
    if (isRunning) {
      stop()
    } else {
      start()
    }
  }

  const handleShareLabels = () => {
    console.log('Sharing labels to:', shareEmail)
    // TODO: Implement actual sharing logic
    setShareModalOpen(false)
    setShareEmail('')
  }

  // Cleanup workers on unmount
  useEffect(() => {
    return () => {
      terminateWorkers()
    }
  }, [terminateWorkers])

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', height: '100vh' }}>
        {/* Mini Sidebar - Always visible when main sidebar is closed */}
        {!sidebarOpen && !isMobile && (
          <Box
            sx={{
              width: miniDrawerWidth,
              flexShrink: 0,
              position: 'fixed',
              height: '100vh',
              zIndex: 1200,
              bgcolor: 'background.paper',
              borderRight: 1,
              borderColor: 'divider',
            }}
          >
            <Toolbar sx={{ justifyContent: 'center' }}>
              <IconButton onClick={handleDrawerToggle} size="large">
                <MenuIcon />
              </IconButton>
            </Toolbar>
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', p: 1 }}>
              <IconButton
                size="large"
                sx={{ mb: 2 }}
                onClick={handleRunStopClick}
                disabled={!selectedFile || isLoadingData}
              >
                {isRunning ? <StopIcon /> : <PlayArrowIcon />}
              </IconButton>
            </Box>
          </Box>
        )}

        {/* Main Sidebar */}
        <Drawer
          variant={isMobile ? 'temporary' : 'persistent'}
          open={sidebarOpen}
          onClose={handleDrawerToggle}
          sx={{
            width: sidebarOpen ? drawerWidth : 0,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: drawerWidth,
              boxSizing: 'border-box',
            },
          }}
        >
          <Toolbar>
            <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
              Cytoverse
            </Typography>
            <IconButton sx={{ mr: 1 }} size="small" onClick={() => setHelpModalOpen(true)}>
              <HelpOutlineIcon />
            </IconButton>
            <IconButton
              sx={{ mr: 1 }}
              size="small"
              href="https://github.com/braingeneers/cytoverse"
              component="a"
            >
              <GitHubIcon />
            </IconButton>
            <IconButton onClick={handleDrawerToggle}>
              <ChevronLeftIcon />
            </IconButton>
          </Toolbar>
          <Box sx={{ overflow: 'auto', p: 2 }}>
            {/* File Selection */}
            <Box sx={{ mb: 3 }}>
              <MuiFileInput
                key={selectedFile?.name || 'no-file'}
                value={selectedFile}
                onChange={setSelectedFile}
                label="File"
                variant="outlined"
                fullWidth
                inputProps={{ accept: '.h5ad' }}
                placeholder="Select a .h5ad file"
              />
            </Box>

            {/* Model Selection */}
            <Box sx={{ mb: 3 }}>
              <FormControl fullWidth>
                <InputLabel>Reference</InputLabel>
                <Select
                  value={selectedModel}
                  label="Reference"
                  onChange={(e) => setSelectedModel(e.target.value)}
                  disabled={isRunning}
                >
                  <MenuItem value="brain">Brain</MenuItem>
                  <MenuItem value="scimilarity">SCimilarity</MenuItem>
                </Select>
              </FormControl>
            </Box>

            {/* Category Selection */}
            <Box sx={{ mb: 3 }}>
              <FormControl fullWidth>
                <InputLabel>Category</InputLabel>
                <Select
                  value={selectedCategory}
                  label="Category"
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  disabled={isRunning}
                >
                  {availableCategories.map((category) => (
                    <MenuItem key={category} value={category}>
                      {category}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>

            {/* Dataset Statistics */}
            <Box sx={{ mt: 3, p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
              {xTrainData && yTrainData && categoryLabels && categoryData ? (
                <Box>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Reference Cells:</strong> {categoryData.length.toLocaleString()}
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Plotted Cells:</strong> {xTrainData.length.toLocaleString()}
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Labels:</strong> {categoryLabels.length}
                  </Typography>
                </Box>
              ) : isLoadingData ? (
                <Typography variant="body2" color="text.secondary">
                  Loading statistics...
                </Typography>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No data loaded
                </Typography>
              )}
            </Box>

            {/* WebGPU Selection */}
            <Box sx={{ mb: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <FormControl component="fieldset" disabled={isRunning} sx={{ flexGrow: 1 }}>
                  <RadioGroup
                    value={useWebGPU ? 'gpu' : 'cpu'}
                    onChange={(e) => setUseWebGPU(e.target.value === 'gpu')}
                    row
                  >
                    <FormControlLabel
                      data-testid="radio-cpu-option"
                      value="cpu"
                      control={<Radio />}
                      label="CPU"
                    />
                    <FormControlLabel
                      data-testid="radio-gpu-option"
                      value="gpu"
                      control={<Radio />}
                      label="GPU"
                      disabled={!hasWebGPU}
                    />
                  </RadioGroup>
                </FormControl>
                {!hasWebGPU && (
                  <Tooltip title="WebGPU not available in this browser">
                    <InfoIcon color="disabled" fontSize="small" />
                  </Tooltip>
                )}
              </Box>
            </Box>

            {/* Run/Stop Button */}
            <Button
              data-testid="run-stop-button"
              variant="contained"
              size="large"
              fullWidth
              startIcon={isRunning ? <StopIcon /> : <PlayArrowIcon />}
              onClick={handleRunStopClick}
              disabled={!selectedFile || isLoadingData}
              sx={{ mb: 3 }}
            >
              {isRunning ? 'Stop' : 'Start'}
            </Button>

            {/* Progress and Status */}
            {(progress > 0 || statusMessage) && (
              <Box sx={{ mb: 3 }}>
                <Typography data-testid="status" variant="body2" sx={{ mb: 1 }}>
                  {statusMessage}
                </Typography>
                <LinearProgress variant="determinate" value={progress} />
                <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
                  {progress.toFixed(1)}%
                </Typography>
              </Box>
            )}

            {/* Label Counts */}
            {Object.keys(labelCounts).length > 0 && (
              <Box sx={{ mb: 3 }}>
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    mb: 2,
                  }}
                >
                  <Typography variant="h6" sx={{ flexGrow: 1 }}>
                    {selectedCategory && `${selectedCategory}`}
                  </Typography>
                  <IconButton
                    data-testid="download-button"
                    sx={{ mr: 1 }}
                    size="small"
                    onClick={exportResultsToCSV}
                    disabled={Object.keys(labelCounts).length === 0 || isRunning}
                    title="Download results as CSV"
                  >
                    <DownloadIcon />
                  </IconButton>
                  <IconButton
                    sx={{ mr: 1 }}
                    size="small"
                    onClick={() => setShareModalOpen(true)}
                    disabled={Object.keys(labelCounts).length === 0 || isRunning}
                  >
                    <ShareIcon />
                  </IconButton>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body1">Total:</Typography>
                  <Typography variant="body1">{totalLabeled.toLocaleString()}</Typography>
                </Box>
                {Object.entries(labelCounts)
                  .sort(([, a], [, b]) => b - a)
                  .map(([label, count]) => (
                    <Box
                      key={label}
                      sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}
                    >
                      <Typography variant="body2" noWrap sx={{ flexGrow: 1, mr: 1 }}>
                        {label}
                      </Typography>
                      <Typography variant="body2">{count.toLocaleString()}</Typography>
                    </Box>
                  ))}
              </Box>
            )}
          </Box>
        </Drawer>

        {/* Main Content */}
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            marginLeft: sidebarOpen ? 0 : isMobile ? 0 : `${miniDrawerWidth}px`,
            transition: (theme) =>
              theme.transitions.create('margin', {
                easing: theme.transitions.easing.sharp,
                duration: theme.transitions.duration.leavingScreen,
              }),
          }}
        >
          {isLoadingData ? (
            <Box display="flex" justifyContent="center" alignItems="center" height="100%">
              <Typography>Loading scatter plot data...</Typography>
            </Box>
          ) : xTrainData && yTrainData && categoryData && categoryLabels.length > 0 ? (
            <ScatterPlotWebGL
              xTrainData={xTrainData}
              yTrainData={yTrainData}
              categoryData={categoryData}
              categoryLabels={categoryLabels}
              xTestData={xTestData}
              yTestData={yTestData}
              testDataLabels={testDataLabels}
            />
          ) : (
            <Box display="flex" justifyContent="center" alignItems="center" height="100%">
              <Typography>No data available</Typography>
            </Box>
          )}
        </Box>

        {/* Share Modal */}
        <Dialog
          open={shareModalOpen}
          onClose={() => setShareModalOpen(false)}
          maxWidth="sm"
          fullWidth
        >
          <DialogTitle>Share Embeddings</DialogTitle>
          <DialogContent>
            <Box sx={{ pt: 1 }}>
              <TextField
                fullWidth
                label="Email Address"
                type="email"
                value={shareEmail}
                onChange={(e) => setShareEmail(e.target.value)}
                sx={{ mb: 2 }}
              />
              <Typography variant="body2" color="text.secondary">
                Get introduced to others with similar embeddings? Enter your email and we'll connect
                you. We'll never upload or share your raw data, only your embeddings. (Coming
                soon...)
              </Typography>
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setShareModalOpen(false)}>Cancel</Button>
            <Button variant="contained" onClick={handleShareLabels} disabled={!shareEmail.trim()}>
              Share
            </Button>
          </DialogActions>
        </Dialog>

        {/* Help Modal */}
        <Dialog open={helpModalOpen} onClose={() => setHelpModalOpen(true)}>
          <DialogTitle>Help</DialogTitle>
          <DialogContent>
            <Typography paragraph>
              CytoVerse is a browser-based platform for single-cell RNA-seq analysis, designed for
              cell annotation using foundation model embeddings. It runs entirely in the browser,
              streaming h5ad files from local storage without uploading data or requiring server
              computation. It uses SCimilarity for cell embeddings and parametric UMAP for 2D
              visualization, leveraging large training datasets for accurate cell annotation. An
              Inverted File with Product Quantization (IVFPQ) enables fast approximate nearest
              neighbor searches across over 20 million samples. Built on WebAssembly and ONNX for
              high-speed processing, it supports unlimited streaming analysis of h5ad files via
              h5wasm. This enables distributed collaborative discovery, allowing researchers to
              explore shared embedding spaces to identify overlapping or complementary assays,
              particularly for perturbseq-driven research. The architecture ensures privacy,
              scalability, and collaborative potential without server dependency.
            </Typography>
            <Typography paragraph>
              <strong>Getting Started:</strong>
            </Typography>
            <Typography component="div">
              <ol>
                <li>Select an H5AD file containing single-cell RNA-seq data</li>
                <li>Choose a pre-trained model (Brain or Scimilarity)</li>
                <li>Select a category for cell type prediction</li>
                <li>Choose CPU or GPU processing (if available)</li>
                <li>Click Start to begin analysis</li>
              </ol>
            </Typography>
            <Typography paragraph>
              The tool will display your cells in real-time as they are processed and labeled.
            </Typography>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setHelpModalOpen(false)}>Close</Button>
          </DialogActions>
        </Dialog>

        {/* Error Modal */}
        <Dialog open={errorModalOpen} onClose={() => setErrorModalOpen(false)}>
          <DialogTitle>Error</DialogTitle>
          <DialogContent>
            <Typography data-testid="error-title">{errorMessage}</Typography>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setErrorModalOpen(false)}>Close</Button>
          </DialogActions>
        </Dialog>
      </Box>
    </ThemeProvider>
  )
}

export default App

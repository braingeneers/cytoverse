import { useEffect, useState, useCallback } from 'react'
import { ThemeProvider } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'
import { createTheme } from '@mui/material/styles'
import {
  Alert,
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
} from '@mui/material'
import MenuIcon from '@mui/icons-material/Menu'
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import StopIcon from '@mui/icons-material/Stop'
import { MuiFileInput } from 'mui-file-input'
import { tableFromIPC, Vector } from 'apache-arrow'

import EmbeddingWorker from './embedder?worker'
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

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [statusMessage, setStatusMessage] = useState('')
  const [progress, setProgress] = useState(0)

  const [embedderWorker, setEmbedderWorker] = useState<Worker | null>(null)
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
  const [selectedCategory, setSelectedCategory] = useState<string>('tissue') // TODO: Add category selector UI

  // Test data state - incremental mappings
  const [xTestData, setXTestData] = useState<number[]>([])
  const [yTestData, setYTestData] = useState<number[]>([])
  const [scatterPlotRef, setScatterPlotRef] = useState<{
    drawNewPoints: (x: number[], y: number[]) => void
  } | null>(null)

  // Normalization parameters from metadata
  const [xCenter, setXCenter] = useState<number>(0)
  const [yCenter, setYCenter] = useState<number>(0)
  const [maxRange, setMaxRange] = useState<number>(1)

  const detectWebGPU = useCallback(async () => {
    try {
      const webGPUSupported = await isWebGPUSupported()
      setHasWebGPU(webGPUSupported)
      // Default to GPU if available, otherwise CPU
      setUseWebGPU(webGPUSupported)
    } catch (error) {
      console.error('WebGPU detection failed:', error)
      setHasWebGPU(false)
      setUseWebGPU(false)
    }

    // REMIND: Remove once we have a stable WebGPU implementation
    setHasWebGPU(false)
    setUseWebGPU(false)
  }, [])

  // Function to check WebGPU support
  async function isWebGPUSupported(): Promise<boolean> {
    try {
      // Check if WebGPU is available in the browser
      if (!navigator.gpu) {
        console.log('WebGPU is not supported in this browser.')
        return false
      }

      // Check if a WebGPU adapter is available
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

  function createWorkers() {
    const embedder = new EmbeddingWorker()

    embedder.onmessage = (evt) => {
      switch (evt.data.type) {
        case 'status':
          setStatusMessage(evt.data.message)
          break
        case 'progress':
          setStatusMessage(evt.data.message)
          setProgress(Math.round((evt.data.countFinished / evt.data.totalToProcess) * 100))
          break
        case 'embeddings':
          console.log('Received embedding:', evt.data.embeddings.slice(0, 4), '...')
          break
        case 'mappings':
          console.log('Received mappings batch:', evt.data.mappings.length, 'points')
          // Add new mappings to test data - mappings contains [[x1, y1], [x2, y2], ...]
          if (evt.data.mappings && evt.data.mappings.length > 0) {
            const newXPoints: number[] = []
            const newYPoints: number[] = []

            // Extract X and Y coordinates from the array of tuples and normalize them
            for (const mapping of evt.data.mappings) {
              if (mapping && mapping.length >= 2) {
                // Normalize the coordinates using the same transformation as mapper.py
                const [normalizedX, normalizedY] = normalizeCoordinates(mapping[0], mapping[1])
                newXPoints.push(normalizedX)
                newYPoints.push(normalizedY)
              }
            }

            setXTestData((prev) => [...prev, ...newXPoints])
            setYTestData((prev) => [...prev, ...newYPoints])

            // Draw the new points on the scatter plot if ref is available
            if (scatterPlotRef) {
              scatterPlotRef.drawNewPoints(newXPoints, newYPoints)
            }

            console.log('Added', newXPoints.length, 'new normalized test points')
          }
          break
        case 'finished':
          setStatusMessage(
            `Processed ${evt.data.totalProcessed} of ${
              evt.data.totalNumCells
            } cells in ${evt.data.elapsedTime?.toFixed(2)} minutes`
          )
          setIsRunning(false)
          break
        case 'error':
          setStatusMessage(evt.data.error.toString())
          setIsRunning(false)
          break
        default:
          break
      }
    }
    return embedder
  }

  const sitePath =
    window.location.origin +
    window.location.pathname.slice(0, window.location.pathname.lastIndexOf('/'))

  // Helper function to normalize coordinates using the same transformation as mapper.py
  const normalizeCoordinates = (x: number, y: number) => {
    const normalizedX = (x - xCenter) / (maxRange / 2)
    const normalizedY = (y - yCenter) / (maxRange / 2)
    return [normalizedX, normalizedY]
  }

  const loadTrainingData = useCallback(async () => {
    setIsLoadingData(true)
    try {
      const modelID = 'scimilarity'
      setSelectedCategory('tissue') // Default category, can be changed later

      // Load metadata to get categories information
      const metadataResponse = await fetch(`${sitePath}/models/${modelID}/metadata.json`)
      const metadata = await metadataResponse.json()

      // Get category labels from metadata
      const labels = metadata.categories?.[selectedCategory]
      if (!labels || !Array.isArray(labels)) {
        throw new Error(`Missing or invalid category labels for '${selectedCategory}' in metadata`)
      }

      // Store normalization parameters for test data
      setXCenter(metadata.xCenter || 0)
      setYCenter(metadata.yCenter || 0)
      setMaxRange(metadata.maxRange || 1)

      // Load Arrow files in parallel
      const [xResponse, yResponse, categoryResponse] = await Promise.all([
        fetch(`${sitePath}/models/${modelID}/x.arrow`),
        fetch(`${sitePath}/models/${modelID}/y.arrow`),
        fetch(`${sitePath}/models/${modelID}/${selectedCategory}.arrow`),
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

      // Extract columns
      const xColumn = xTable.getChild('x')
      const yColumn = yTable.getChild('y')
      const categoryColumn = categoryTable.getChild(selectedCategory)

      if (!xColumn || !yColumn || !categoryColumn) {
        throw new Error('Missing required columns in Arrow files')
      }

      const numPoints = xTable.numRows

      // Validate all tables have the same number of rows
      if (yTable.numRows !== numPoints || categoryTable.numRows !== numPoints) {
        throw new Error('Mismatched number of rows between Arrow files')
      }

      // Set the Vector objects directly - no need for data transformation
      setXTrainData(xColumn)
      setYTrainData(yColumn)
      setCategoryData(categoryColumn)
      setCategoryLabels(labels)

      // console.log('Loaded scatter plot data:', {
      //   numPoints,
      //   numCategories: labels.length,
      //   category: selectedCategory,
      //   xRange: [
      //     Math.min(...Array.from({ length: xColumn.length }, (_, i) => xColumn.get(i) || 0)),
      //     Math.max(...Array.from({ length: xColumn.length }, (_, i) => xColumn.get(i) || 0)),
      //   ],
      //   yRange: [
      //     Math.min(...Array.from({ length: yColumn.length }, (_, i) => yColumn.get(i) || 0)),
      //     Math.max(...Array.from({ length: yColumn.length }, (_, i) => yColumn.get(i) || 0)),
      //   ],
      //   categoryRange: [
      //     Math.min(
      //       ...Array.from({ length: categoryColumn.length }, (_, i) => categoryColumn.get(i) || 0)
      //     ),
      //     Math.max(
      //       ...Array.from({ length: categoryColumn.length }, (_, i) => categoryColumn.get(i) || 0)
      //     ),
      //   ],
      //   sampleLabels: labels.slice(0, 10), // Show first 10 for debugging
      // })
    } catch (error) {
      console.error('Error loading scatter plot data:', error)
    } finally {
      setIsLoadingData(false)
    }
  }, [sitePath, selectedCategory])

  useEffect(() => {
    console.log('App mounted')
    fetchSampleFile()
    detectWebGPU()
    loadTrainingData()
  }, [detectWebGPU, loadTrainingData])

  const start = () => {
    console.log('Starting embedding...', selectedFile?.name)
    setProgress(0)
    setIsRunning(true)

    // Clear any existing test data
    setXTestData([])
    setYTestData([])

    const embedder = createWorkers()
    setEmbedderWorker(embedder)
    embedder.postMessage({
      type: 'start',
      modelsURL: `${sitePath}/models`,
      modelID: 'scimilarity',
      h5File: selectedFile,
      cellRangePercent: 100,
      useWebGPU: useWebGPU,
    })
  }

  const stop = () => {
    setStatusMessage('Stopping processing...')
    setIsRunning(false)
    setProgress(0)

    // // Clear test data
    // setXTestData([])
    // setYTestData([])

    if (embedderWorker) {
      embedderWorker.terminate()
      setEmbedderWorker(null)
    }
    setStatusMessage('Processing stopped')
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

  useEffect(() => {
    return () => {
      if (embedderWorker) {
        console.log('Terminating worker...')
        embedderWorker.terminate()
        setEmbedderWorker(null)
        console.log('Worker terminated')
      }
    }
  }, [embedderWorker])

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
              backgroundColor: 'background.paper',
              borderRight: '1px solid',
              borderColor: 'divider',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              pt: 2,
              pb: 2,
            }}
          >
            <IconButton
              color="primary"
              aria-label="open drawer"
              onClick={handleDrawerToggle}
              sx={{ mb: 2 }}
            >
              <MenuIcon />
            </IconButton>
            <IconButton
              color={isRunning ? 'error' : 'primary'}
              disabled={!selectedFile}
              onClick={handleRunStopClick}
              sx={{
                width: 48,
                height: 48,
                backgroundColor: isRunning
                  ? 'error.main'
                  : selectedFile
                  ? 'primary.main'
                  : 'action.disabled',
                color: isRunning
                  ? 'error.contrastText'
                  : selectedFile
                  ? 'primary.contrastText'
                  : 'text.disabled',
                '&:hover': {
                  backgroundColor: isRunning
                    ? 'error.dark'
                    : selectedFile
                    ? 'primary.dark'
                    : 'action.hover',
                },
                '&:disabled': {
                  backgroundColor: 'action.disabled',
                  color: 'text.disabled',
                },
                '& .MuiSvgIcon-root': {
                  fontSize: '1.5rem',
                },
              }}
            >
              {isRunning ? <StopIcon /> : <PlayArrowIcon />}
            </IconButton>
          </Box>
        )}

        {/* Hamburger Menu Button - Only shown on mobile when sidebar is closed */}
        {!sidebarOpen && isMobile && (
          <IconButton
            color="primary"
            aria-label="open drawer"
            onClick={handleDrawerToggle}
            sx={{
              position: 'fixed',
              top: 16,
              left: 16,
              zIndex: (theme) => theme.zIndex.drawer + 1,
              backgroundColor: 'background.paper',
              '&:hover': {
                backgroundColor: 'action.hover',
              },
            }}
          >
            <MenuIcon />
          </IconButton>
        )}

        {/* Sidebar Drawer */}
        <Drawer
          variant={isMobile ? 'temporary' : 'persistent'}
          open={sidebarOpen}
          onClose={isMobile ? handleDrawerToggle : undefined}
          sx={{
            width: drawerWidth,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: drawerWidth,
              boxSizing: 'border-box',
            },
          }}
        >
          <Toolbar>
            <Typography variant="h6" sx={{ flexGrow: 1 }}>
              CellSpace
            </Typography>
            <IconButton onClick={handleDrawerToggle}>
              <ChevronLeftIcon />
            </IconButton>
          </Toolbar>
          <Box sx={{ p: 2 }}>
            <MuiFileInput
              fullWidth
              placeholder="Select an AnnData/Scanpy (.h5ad) file"
              value={selectedFile}
              onChange={setSelectedFile}
              inputProps={{ accept: '.h5ad' }}
              sx={{ mb: 2 }}
            />

            {/* Execution Provider Selection */}
            <FormControl component="fieldset" sx={{ mb: 2 }}>
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

            <Button
              data-testid="run-stop-button"
              variant="contained"
              fullWidth
              disabled={!selectedFile}
              onClick={handleRunStopClick}
              startIcon={isRunning ? <StopIcon /> : <PlayArrowIcon />}
              sx={{
                height: 56,
                borderRadius: 1,
                backgroundColor: isRunning ? 'error.main' : 'primary.main',
                '&:hover': {
                  backgroundColor: isRunning ? 'error.dark' : 'primary.dark',
                },
                '&:disabled': {
                  backgroundColor: 'action.disabled',
                  color: 'text.disabled',
                },
                fontSize: '1rem',
                fontWeight: 'bold',
              }}
            >
              {isRunning ? 'STOP' : 'RUN'}
            </Button>

            {/* Dataset Statistics */}
            <Box sx={{ mt: 3, p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
              {xTrainData && yTrainData && categoryData ? (
                <Box>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Cells:</strong> {xTrainData.length.toLocaleString()}
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Category:</strong> {selectedCategory}
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
          </Box>

          {/* Status and Progress */}
          <Typography>{statusMessage}</Typography>
          {isRunning && (
            <Box my={2}>
              <LinearProgress variant="determinate" value={progress} />
              <Typography>{progress}%</Typography>
            </Box>
          )}

          {!window.crossOriginIsolated && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              Unable to use multiple cpu cores - notify the site owner
            </Alert>
          )}
        </Drawer>

        {/* Main Content */}
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            width: {
              sm: `calc(100% - ${sidebarOpen ? drawerWidth : isMobile ? 0 : miniDrawerWidth}px)`,
            },
            ml: { sm: sidebarOpen ? 0 : `${miniDrawerWidth}px` },
            transition: (theme) =>
              theme.transitions.create(['margin', 'width'], {
                easing: theme.transitions.easing.sharp,
                duration: theme.transitions.duration.leavingScreen,
              }),
            height: '100vh',
            display: 'flex',
            flexDirection: 'column',
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
              xTestData={xTestData}
              yTestData={yTestData}
              categoryData={categoryData}
              categoryLabels={categoryLabels}
              onRef={setScatterPlotRef}
            />
          ) : (
            <Box display="flex" justifyContent="center" alignItems="center" height="100%">
              <Typography>No data available</Typography>
            </Box>
          )}
        </Box>
      </Box>
    </ThemeProvider>
  )
}

export default App

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
  Link,
  useMediaQuery,
  Button,
  FormControl,
  RadioGroup,
  FormControlLabel,
  Radio,
  Checkbox,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
} from '@mui/material'
import MenuIcon from '@mui/icons-material/Menu'
import GitHubIcon from '@mui/icons-material/GitHub'
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import StopIcon from '@mui/icons-material/Stop'
import ShareIcon from '@mui/icons-material/Share'
import HelpOutlineIcon from '@mui/icons-material/HelpOutline'
import { MuiFileInput } from 'mui-file-input'
import { tableFromIPC, Vector } from 'apache-arrow'

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

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [statusMessage, setStatusMessage] = useState('')
  const [progress, setProgress] = useState(0)

  const [embedderWorker, setEmbedderWorker] = useState<Worker | null>(null)
  const [labelerWorker, setLabelerWorker] = useState<Worker | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [hasWebGPU, setHasWebGPU] = useState(false)
  const [useWebGPU, setUseWebGPU] = useState(false)
  const [labelingEnabled, setLabelingEnabled] = useState(true) // Default to enabled
  const isMobile = useMediaQuery(theme.breakpoints.down('md'))

  // Scatter plot data state - training data (static)
  const [xTrainData, setXTrainData] = useState<Vector | null>(null)
  const [yTrainData, setYTrainData] = useState<Vector | null>(null)
  const [categoryData, setCategoryData] = useState<Vector | null>(null)
  const [categoryLabels, setCategoryLabels] = useState<string[]>([])
  const [isLoadingData, setIsLoadingData] = useState(false)
  const [selectedCategory, setSelectedCategory] = useState<string>('prediction') // TODO: Add category selector UI

  // Test data state - incremental mappings
  const [xTestData, setXTestData] = useState<number[]>([])
  const [yTestData, setYTestData] = useState<number[]>([])

  // Labeling feedback state - counts of predicted labels
  const [labelCounts, setLabelCounts] = useState<{ [label: string]: number }>({})
  const [totalLabeled, setTotalLabeled] = useState<number>(0)

  // Share modal state
  const [shareModalOpen, setShareModalOpen] = useState(false)
  const [shareEmail, setShareEmail] = useState('')

  // Help modal state
  const [helpModalOpen, setHelpModalOpen] = useState(false)

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
    console.warn('WebGPU is currently disabled as Apple silicon limits input vectors to 16384')
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
    let labeler: Worker | null = null

    // Create labeler worker if labeling is enabled
    if (labelingEnabled) {
      labeler = new LabelerWorker()
      setLabelerWorker(labeler)

      // Initialize labeler
      labeler.postMessage({
        type: 'start',
        modelsURL: `${sitePath}/models`,
        modelID: 'scimilarity',
      })

      // Handle labeler messages
      labeler.onmessage = (evt) => {
        switch (evt.data.type) {
          case 'status':
            console.log('Labeler status:', evt.data.message)
            break
          case 'labeled':
            console.log('Received labeled batch:', evt.data.umap_coordinates?.length, 'points')
            // Add new embeddings to test data - umap_coordinates contains [[x1, y1], [x2, y2], ...]
            if (evt.data.umap_coordinates && evt.data.umap_coordinates.length > 0) {
              const newXPoints: number[] = []
              const newYPoints: number[] = []

              // Extract X and Y coordinates from the array of tuples and normalize them
              for (const coordinate of evt.data.umap_coordinates) {
                if (coordinate && coordinate.length >= 2) {
                  // Normalize the coordinates using the same transformation as pumap.py
                  const [normalizedX, normalizedY] = normalizeCoordinates(
                    coordinate[0],
                    coordinate[1]
                  )
                  newXPoints.push(normalizedX)
                  newYPoints.push(normalizedY)
                }
              }

              setXTestData((prev) => [...prev, ...newXPoints])
              setYTestData((prev) => [...prev, ...newYPoints])

              console.log('Added', newXPoints.length, 'new labeled test points')
            }

            // Process labeling results for feedback tallies
            if (evt.data.train_vector_id && categoryData && categoryLabels.length > 0) {
              const newLabelCounts: { [label: string]: number } = {}
              let validLabels = 0

              for (const trainVectorId of evt.data.train_vector_id) {
                if (trainVectorId !== -1 && trainVectorId < categoryData.length) {
                  // Get category index from training data
                  const categoryIndex = categoryData.get(trainVectorId)
                  if (categoryIndex != null && categoryIndex < categoryLabels.length) {
                    const labelName = categoryLabels[categoryIndex]
                    newLabelCounts[labelName] = (newLabelCounts[labelName] || 0) + 1
                    validLabels++
                  }
                }
              }

              // Update label counts
              setLabelCounts((prev) => {
                const updated = { ...prev }
                for (const [label, count] of Object.entries(newLabelCounts)) {
                  updated[label] = (updated[label] || 0) + count
                }
                return updated
              })

              setTotalLabeled((prev) => prev + validLabels)
            }
            break
          case 'error':
            console.error('Labeler error:', evt.data.error)
            setStatusMessage(`Labeling error: ${evt.data.error}`)
            break
          default:
            break
        }
      }
    }

    embedder.onmessage = (evt) => {
      switch (evt.data.type) {
        case 'status':
          setStatusMessage(evt.data.message)
          break
        case 'progress':
          setStatusMessage(evt.data.message)
          setProgress(Math.round((evt.data.countFinished / evt.data.totalToProcess) * 100))
          break
        case 'embedding':
          console.log('Received embedding batch:', evt.data.umap_coordinates?.length, 'points')

          if (labelingEnabled && labeler) {
            // Route through labeler for cell type prediction
            labeler.postMessage({
              type: 'embedding',
              test_vector_id: evt.data.test_vector_id,
              pq_embedding: evt.data.pq_embedding,
              umap_coordinates: evt.data.umap_coordinates,
            })
          } else {
            // Direct visualization without labeling
            if (evt.data.umap_coordinates && evt.data.umap_coordinates.length > 0) {
              const newXPoints: number[] = []
              const newYPoints: number[] = []

              // Extract X and Y coordinates from the array of tuples and normalize them
              for (const coordinate of evt.data.umap_coordinates) {
                if (coordinate && coordinate.length >= 2) {
                  // Normalize the coordinates using the same transformation as pumap.py
                  const [normalizedX, normalizedY] = normalizeCoordinates(
                    coordinate[0],
                    coordinate[1]
                  )
                  newXPoints.push(normalizedX)
                  newYPoints.push(normalizedY)
                }
              }

              setXTestData((prev) => [...prev, ...newXPoints])
              setYTestData((prev) => [...prev, ...newYPoints])

              console.log('Added', newXPoints.length, 'new normalized test points')
            }
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

  // Helper function to normalize coordinates using the same transformation as pumap.py
  const normalizeCoordinates = (x: number, y: number) => {
    const normalizedX = (x - xCenter) / (maxRange / 2)
    const normalizedY = (y - yCenter) / (maxRange / 2)
    return [normalizedX, normalizedY]
  }

  const loadTrainingData = useCallback(async () => {
    setIsLoadingData(true)
    try {
      const modelID = 'scimilarity'
      setSelectedCategory('prediction') // Default category, can be changed later

      // Load metadata to get categories information
      const metadataResponse = await fetch(`${sitePath}/models/${modelID}/pumap/metadata.json`)
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

      // Extract columns
      const xColumn = xTable.getChild('x')
      const yColumn = yTable.getChild('y')
      const categoryColumn = categoryTable.getChild(selectedCategory)

      if (!xColumn || !yColumn || !categoryColumn) {
        throw new Error('Missing required columns in Arrow files')
      }

      // Set the Vector objects directly - no need for data transformation
      setXTrainData(xColumn)
      setYTrainData(yColumn)
      setCategoryData(categoryColumn)
      setCategoryLabels(labels)
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

    // Clear any existing test data and label counts
    setXTestData([])
    setYTestData([])
    setLabelCounts({})
    setTotalLabeled(0)

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

    if (embedderWorker) {
      embedderWorker.terminate()
      setEmbedderWorker(null)
    }
    if (labelerWorker) {
      labelerWorker.terminate()
      setLabelerWorker(null)
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

  const handleShareLabels = () => {
    console.log('Sharing labels to:', shareEmail)
    // TODO: Implement actual sharing logic
    setShareModalOpen(false)
    setShareEmail('')
  }

  useEffect(() => {
    return () => {
      if (embedderWorker) {
        console.log('Terminating embedder worker...')
        embedderWorker.terminate()
        setEmbedderWorker(null)
        console.log('Embedder worker terminated')
      }
      if (labelerWorker) {
        console.log('Terminating labeler worker...')
        labelerWorker.terminate()
        setLabelerWorker(null)
        console.log('Labeler worker terminated')
      }
    }
  }, [embedderWorker, labelerWorker])

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
              CytoVerse
            </Typography>
            <IconButton
              size="small"
              onClick={() => setHelpModalOpen(true)}
              sx={{
                color: 'text.secondary',
                mr: 1,
                '&:focus': {
                  outline: 'none',
                },
                '&:focus-visible': {
                  outline: 'none',
                },
              }}
            >
              <HelpOutlineIcon />
            </IconButton>
            <IconButton
              component={Link}
              href="https://github.com/braingeneers/cytoverse"
              target="_blank"
              sx={{
                color: 'inherit',
                mr: 1,
                '&:focus': {
                  outline: 'none',
                },
                '&:focus-visible': {
                  outline: 'none',
                },
              }}
            >
              <GitHubIcon />
            </IconButton>
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

            {/* Labeling Toggle */}
            <FormControl component="fieldset" sx={{ mb: 2 }}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={labelingEnabled}
                    onChange={(e) => setLabelingEnabled(e.target.checked)}
                    color="primary"
                  />
                }
                label="Enable real-time cell labeling"
              />
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
              {xTrainData && yTrainData && categoryLabels && categoryData ? (
                <Box>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Plotted Cells:</strong> {xTrainData.length.toLocaleString()}
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Category:</strong> {selectedCategory}
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Labels:</strong> {categoryLabels.length}
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Reference Cells:</strong> {categoryData.length.toLocaleString()}
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

            {/* Status and Progress */}
            <Box sx={{ mt: 2 }}>
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
            </Box>

            {/* Predicted Labels */}
            {labelingEnabled && (
              <Box
                sx={{
                  mt: 2,
                  p: 2,
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 1,
                }}
              >
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    mb: 2,
                  }}
                >
                  <Typography variant="h6">Predicted Labels</Typography>
                  <IconButton
                    size="small"
                    onClick={() => setShareModalOpen(true)}
                    sx={{
                      color: 'primary.main',
                      '&:focus': {
                        outline: 'none',
                      },
                      '&:focus-visible': {
                        outline: 'none',
                      },
                    }}
                  >
                    <ShareIcon />
                  </IconButton>
                </Box>
                {totalLabeled > 0 ? (
                  <Box>
                    <Typography variant="body2" sx={{ mb: 2, fontWeight: 'bold' }}>
                      Total Labeled: {totalLabeled.toLocaleString()}
                    </Typography>
                    <Box
                      sx={{
                        maxHeight: 200,
                        overflowY: 'auto',
                        // Hide scrollbar while keeping functionality
                        '&::-webkit-scrollbar': {
                          display: 'none',
                        },
                        '-ms-overflow-style': 'none',
                        'scrollbar-width': 'none',
                      }}
                    >
                      {Object.entries(labelCounts)
                        .sort(([, a], [, b]) => b - a) // Sort by count descending
                        .map(([label, count]) => (
                          <Box
                            key={label}
                            sx={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center',
                              py: 0.5,
                              borderBottom: '1px solid',
                              borderColor: 'divider',
                            }}
                          >
                            <Typography
                              variant="body2"
                              sx={{
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap',
                                flex: 1,
                                mr: 1,
                              }}
                            >
                              {label}
                            </Typography>
                            <Typography
                              variant="body2"
                              sx={{
                                fontWeight: 'bold',
                                minWidth: 'fit-content',
                              }}
                            >
                              {count.toLocaleString()}
                            </Typography>
                          </Box>
                        ))}
                    </Box>
                  </Box>
                ) : isRunning ? (
                  <Typography variant="body2" color="text.secondary">
                    Waiting for predictions...
                  </Typography>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No predictions yet
                  </Typography>
                )}
              </Box>
            )}
          </Box>
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
            />
          ) : (
            <Box display="flex" justifyContent="center" alignItems="center" height="100%">
              <Typography>No data available</Typography>
            </Box>
          )}
        </Box>
      </Box>

      {/* Share Modal */}
      <Dialog
        open={shareModalOpen}
        onClose={() => setShareModalOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Share Predicted Labels</DialogTitle>
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
              you. We'll never upload or share your raw data, only your embeddings. (feature not yet
              live...)
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
      <Dialog open={helpModalOpen} onClose={() => setHelpModalOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>About CytoVerse</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 1 }}>
            <Typography variant="body1" sx={{ mb: 2 }}>
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
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setHelpModalOpen(false)} variant="contained">
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </ThemeProvider>
  )
}

export default App

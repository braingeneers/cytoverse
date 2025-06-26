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

import EmbeddingWorker from './embedder?worker'

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

  useEffect(() => {
    console.log('App mounted')
    fetchSampleFile()
    detectWebGPU()
  }, [detectWebGPU])

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
          console.log('Received mappings:', evt.data.mappings.slice(0, 4), '...')
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

  const start = () => {
    console.log('Starting embedding...', selectedFile?.name)
    setProgress(0)
    setIsRunning(true)
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
      </Box>
    </ThemeProvider>
  )
}

export default App

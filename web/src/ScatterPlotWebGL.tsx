import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react'
import * as echarts from 'echarts'
import { ScatterGLChart } from 'echarts-gl/charts'
import { GridComponent, VisualMapComponent, LegendComponent } from 'echarts/components'

// Register necessary components
echarts.use([ScatterGLChart, GridComponent, VisualMapComponent, LegendComponent])

// Custom color palette
const customColors = [
  // first 12 colours generated with:
  // RColorBrewer::brewer.pal(n = 12, name = "Paired")
  '#A6CEE3',
  '#1F78B4',
  '#B2DF8A',
  '#33A02C',
  '#FB9A99',
  '#E31A1C',
  '#FDBF6F',
  '#FF7F00',
  '#CAB2D6',
  '#6A3D9A',
  '#FFFF99',
  '#B15928',
  // vivid interlude
  '#1ff8ff', // a bright blue
  // next 8 colours generated with:
  // RColorBrewer::brewer.pal(n = 8, "Dark2")
  '#1B9E77',
  '#D95F02',
  '#7570B3',
  '#E7298A',
  '#66A61E',
  '#E6AB02',
  '#A6761D',
  '#666666',
  // list below generated with iwanthue: all colours soft kmeans 20
  '#4b6a53',
  '#b249d5',
  '#7edc45',
  '#5c47b8',
  '#cfd251',
  '#ff69b4',
  '#69c86c',
  '#cd3e50',
  '#83d5af',
  '#da6130',
  '#5e79b2',
  '#c29545',
  '#532a5a',
  '#5f7b35',
  '#c497cf',
  '#773a27',
  '#7cb9cb',
  '#594e50',
  '#d3c4a8',
  '#c17e7f',
]

// Utility function to calculate extents from scatter plot data
const calculateExtents = (
  data: Float32Array
): { xExtent: [number, number]; yExtent: [number, number] } => {
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity

  for (let i = 0; i < data.length; i += 3) {
    const x = data[i]
    const y = data[i + 1]

    minX = Math.min(minX, x)
    maxX = Math.max(maxX, x)
    minY = Math.min(minY, y)
    maxY = Math.max(maxY, y)
  }

  // Add 5% padding to the bounds
  const xPadding = (maxX - minX) * 0.05
  const yPadding = (maxY - minY) * 0.05

  return {
    xExtent: [minX - xPadding, maxX + xPadding],
    yExtent: [minY - yPadding, maxY + yPadding],
  }
}

interface ScatterPlotWebGLProps {
  trainMappings: Float32Array
  testMappings?: Float32Array
  classNames: string[]
  themeName?: string
}

const ScatterPlotWebGL: React.FC<ScatterPlotWebGLProps> = ({
  trainMappings,
  testMappings,
  classNames,
  themeName = 'light',
}) => {
  const chartContainer = useRef<HTMLDivElement>(null)
  const chartRef = useRef<echarts.ECharts | null>(null)
  const [showBoth, setShowBoth] = useState(true)
  const [showTrainOnly, setShowTrainOnly] = useState(false)
  const [showTestOnly, setShowTestOnly] = useState(false)

  const pieces = useMemo(() => {
    return classNames.map((name, index) => {
      return {
        value: index,
        label: name,
        color: customColors[index % customColors.length],
      }
    })
  }, [classNames])

  // Calculate bounds from training data on first mount
  const calculateBounds = useCallback((data: Float32Array) => {
    const extents = calculateExtents(data)
    return {
      xMin: extents.xExtent[0],
      xMax: extents.xExtent[1],
      yMin: extents.yExtent[0],
      yMax: extents.yExtent[1],
    }
  }, [])

  // Function to initialize or reinitialize the chart
  const getChartOption = useCallback(() => {
    const currentBounds = calculateBounds(trainMappings)
    return {
      backgroundColor: 'transparent', // Let the chart inherit the background color
      grid: {
        left: 150, // Reduced legend space for more chart width
        right: 10,
        top: 10,
        bottom: 10,
        containLabel: false,
      },
      xAxis: {
        show: false,
        min: currentBounds.xMin,
        max: currentBounds.xMax,
        type: 'value',
      },
      yAxis: {
        show: false,
        min: currentBounds.yMin,
        max: currentBounds.yMax,
        type: 'value',
      },
      visualMap: {
        type: 'piecewise',
        dimension: 2,
        pieces: pieces,
        outOfRange: {
          symbolSize: 0,
        },
        show: true,
        orient: 'vertical',
        left: 5,
        top: 'center',
        itemWidth: 12,
        itemHeight: 12,
        textStyle: {
          fontSize: 11,
        },
        calculable: false,
        itemGap: 3,
      },
      animation: false, // Disable animation for better performance
      series: [
        {
          name: 'Reference',
          type: 'scatterGL',
          data: [],
          dimensions: ['x', 'y', 'class'],
          symbolSize: 1,
          itemStyle: {
            opacity: 0.3,
          },
        },
        {
          name: 'Predictions',
          type: 'scatterGL',
          data: [],
          dimensions: ['x', 'y', 'class'],
          symbolSize: 5,
        },
      ],
    }
  }, [trainMappings, pieces, calculateBounds])

  const initChart = useCallback(() => {
    if (!chartContainer.current) return
    if (chartRef.current) {
      chartRef.current.dispose()
    }

    chartRef.current = echarts.init(chartContainer.current, themeName)
    chartRef.current.setOption(getChartOption())

    // Multiple resize calls to ensure proper sizing
    setTimeout(() => {
      if (chartRef.current) {
        chartRef.current.resize()
      }
    }, 0)

    setTimeout(() => {
      if (chartRef.current) {
        chartRef.current.resize()
      }
    }, 50)
  }, [themeName, getChartOption])

  const updateChart = useCallback(() => {
    if (!chartRef.current) return

    const options = chartRef.current.getOption() as echarts.EChartsOption
    const visualMap = Array.isArray(options.visualMap) ? options.visualMap[0] : options.visualMap
    const currentSelection =
      (visualMap as echarts.PiecewiseVisualMapComponentOption)?.selected || {}

    let trainDataToShow: Float32Array | number[][] = new Float32Array()
    if ((showBoth || showTrainOnly) && trainMappings) {
      trainDataToShow = trainMappings
    }

    let testDataToShow: Float32Array | number[][] = new Float32Array()
    if ((showBoth || showTestOnly) && testMappings) {
      testDataToShow = testMappings
    }

    chartRef.current.setOption({
      visualMap: {
        selected: currentSelection,
      },
      series: [
        { name: 'Reference', data: trainDataToShow },
        { name: 'Predictions', data: testDataToShow },
      ],
    })
  }, [showBoth, showTrainOnly, showTestOnly, trainMappings, testMappings])

  const setVisibility = useCallback((mode: 'both' | 'train' | 'test') => {
    setShowBoth(mode === 'both')
    setShowTrainOnly(mode === 'train')
    setShowTestOnly(mode === 'test')
  }, [])

  const toggleAllClasses = useCallback((show: boolean) => {
    if (!chartRef.current) return

    const options = chartRef.current.getOption() as echarts.EChartsOption
    const visualMap = (
      Array.isArray(options.visualMap) ? options.visualMap[0] : options.visualMap
    ) as echarts.PiecewiseVisualMapComponentOption
    const currentSelection = visualMap?.selected || {}

    for (let i = 0; i < visualMap.pieces!.length; i++) {
      currentSelection[i] = show
    }

    chartRef.current.setOption({
      visualMap: {
        selected: currentSelection,
      },
    })
  }, [])

  // Effects
  useEffect(() => {
    if (trainMappings && trainMappings.length > 0) {
      initChart()
    }
    return () => {
      if (chartRef.current) {
        chartRef.current.dispose()
        chartRef.current = null
      }
    }
  }, [initChart, trainMappings])

  useEffect(() => {
    initChart()
  }, [classNames, trainMappings, themeName, initChart])

  useEffect(() => {
    updateChart()
  }, [testMappings, updateChart])

  useEffect(() => {
    updateChart()
  }, [showBoth, showTrainOnly, showTestOnly, updateChart])

  // Add resize handler
  useEffect(() => {
    const handleResize = () => {
      if (chartRef.current) {
        chartRef.current.resize()
      }
    }

    window.addEventListener('resize', handleResize)

    // Use ResizeObserver for more precise container resize detection
    let resizeObserver: ResizeObserver | null = null
    if (chartContainer.current && 'ResizeObserver' in window) {
      resizeObserver = new ResizeObserver(() => {
        handleResize()
      })
      resizeObserver.observe(chartContainer.current)
    }

    // Also trigger resize after mount to ensure proper sizing
    const timeoutId = setTimeout(() => {
      if (chartRef.current) {
        chartRef.current.resize()
      }
    }, 100)

    return () => {
      window.removeEventListener('resize', handleResize)
      if (resizeObserver) {
        resizeObserver.disconnect()
      }
      clearTimeout(timeoutId)
    }
  }, [])

  return (
    <div style={{ width: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="controls-header" style={{ flexShrink: 0, marginBottom: '16px' }}>
        <div className="legend-controls">
          <button onClick={() => toggleAllClasses(true)}>Show All</button>
          <button onClick={() => toggleAllClasses(false)}>Hide All</button>
        </div>
        {trainMappings && trainMappings.length > 0 && testMappings && testMappings.length > 0 && (
          <div className="visibility-controls">
            <button className={showBoth ? 'active' : ''} onClick={() => setVisibility('both')}>
              Both
            </button>
            <button
              className={showTrainOnly ? 'active' : ''}
              onClick={() => setVisibility('train')}
            >
              Training
            </button>
            <button className={showTestOnly ? 'active' : ''} onClick={() => setVisibility('test')}>
              Test
            </button>
          </div>
        )}
      </div>
      <div ref={chartContainer} style={{ width: '100%', height: '600px' }} />
    </div>
  )
}

export default ScatterPlotWebGL

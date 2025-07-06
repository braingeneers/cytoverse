import React, { useRef, useEffect } from 'react'
import createScatterplot from 'regl-scatterplot'
import { Vector } from 'apache-arrow'

// Generate distinct colors for categories
const generateCategoryColors = (numCategories: number): string[] => {
  const colors: string[] = []

  // Use HSL to generate evenly spaced colors, avoiding bright red (hue 0)
  for (let i = 0; i < numCategories; i++) {
    // Skip hue 0 (red) and start from 30 degrees to avoid bright red
    const hue = 30 + (i * 330) / numCategories
    const saturation = 0.7
    const lightness = 0.5

    // Convert HSL to RGB
    const c = (1 - Math.abs(2 * lightness - 1)) * saturation
    const x = c * (1 - Math.abs(((hue / 60) % 2) - 1))
    const m = lightness - c / 2

    let r: number, g: number, b: number

    if (hue >= 0 && hue < 60) {
      r = c
      g = x
      b = 0
    } else if (hue >= 60 && hue < 120) {
      r = x
      g = c
      b = 0
    } else if (hue >= 120 && hue < 180) {
      r = 0
      g = c
      b = x
    } else if (hue >= 180 && hue < 240) {
      r = 0
      g = x
      b = c
    } else if (hue >= 240 && hue < 300) {
      r = x
      g = 0
      b = c
    } else {
      r = c
      g = 0
      b = x
    }

    const rHex = Math.round((r + m) * 255)
      .toString(16)
      .padStart(2, '0')
    const gHex = Math.round((g + m) * 255)
      .toString(16)
      .padStart(2, '0')
    const bHex = Math.round((b + m) * 255)
      .toString(16)
      .padStart(2, '0')

    colors.push(`#${rHex}${gHex}${bHex}`)
  }

  return colors
}

interface ScatterPlotWebGLProps {
  xTrainData: Vector
  yTrainData: Vector
  xTestData: number[]
  yTestData: number[]
  categoryData: Vector
  categoryLabels: string[]
}

const ScatterPlotWebGL: React.FC<ScatterPlotWebGLProps> = ({
  xTrainData,
  yTrainData,
  xTestData,
  yTestData,
  categoryData,
  categoryLabels,
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const scatterplotRef = useRef<ReturnType<typeof createScatterplot> | null>(null)
  const isInitializedRef = useRef<boolean>(false)
  const isDrawingRef = useRef<boolean>(false)

  // Initial setup and training data rendering
  useEffect(() => {
    if (!containerRef.current || !canvasRef.current || !xTrainData || !yTrainData || !categoryData)
      return

    // Generate colors for each category + one extra color for test points
    const categoryColors = generateCategoryColors(categoryLabels.length + 1)

    // Set the last color to be a distinct color for test points (e.g., bright red)
    categoryColors[categoryColors.length - 1] = '#FF0000' // Bright red for test points

    // Create the scatterplot instance
    const scatterplot = createScatterplot({
      canvas: canvasRef.current,
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
      pointSize: [1, 5], // Use a range for point sizes
      // performanceMode: true, // Enable performance mode for better rendering
    })

    // Set up categorical coloring and sizing
    scatterplot.set({
      pointColor: categoryColors,
      colorBy: 'valueA',
      sizeBy: 'valueB',
    })

    scatterplotRef.current = scatterplot

    // Convert Arrow Vectors to typed arrays for regl-scatterplot
    // Try to access the underlying buffer directly if possible, otherwise convert
    const numPoints = Math.min(xTrainData.length, yTrainData.length, categoryData.length)

    // Check if we can access the underlying typed array directly
    let xArray: Float32Array
    let yArray: Float32Array

    // Arrow Vector.data should give us access to the underlying data
    if (xTrainData.data.length > 0 && xTrainData.data[0].values instanceof Float32Array) {
      xArray = xTrainData.data[0].values as Float32Array
    } else {
      // Fallback to copying data
      xArray = new Float32Array(numPoints)
      for (let i = 0; i < numPoints; i++) {
        xArray[i] = xTrainData.get(i) || 0
      }
    }

    if (yTrainData.data.length > 0 && yTrainData.data[0].values instanceof Float32Array) {
      yArray = yTrainData.data[0].values as Float32Array
    } else {
      // Fallback to copying data
      yArray = new Float32Array(numPoints)
      for (let i = 0; i < numPoints; i++) {
        yArray[i] = yTrainData.get(i) || 0
      }
    }

    // Get category indices for valueA
    let categoryArrayData: number[]
    if (categoryData.data.length > 0 && categoryData.data[0].values) {
      categoryArrayData = Array.from(categoryData.data[0].values)
    } else {
      // Fallback to vector access
      categoryArrayData = Array.from(
        { length: categoryData.length },
        (_, i) => categoryData.get(i) || 0
      )
    }

    // Initial render with only training data
    const trainX = Array.from(xArray)
    const trainY = Array.from(yArray)
    const trainSize = new Array(numPoints).fill(0) // Training data uses first size

    const initialColumnData = {
      x: new Float32Array(trainX),
      y: new Float32Array(trainY),
      valueA: categoryArrayData,
      valueB: trainSize,
    }

    // Draw the initial training points
    const drawInitialData = async () => {
      await scatterplot.draw(initialColumnData)
      isInitializedRef.current = true
      console.log(
        'Drew',
        numPoints,
        'training points with',
        categoryColors.length,
        'category colors'
      )
    }
    drawInitialData()

    // Handle resize
    const handleResize = () => {
      if (containerRef.current && scatterplot) {
        scatterplot.set({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        })
      }
    }

    window.addEventListener('resize', handleResize)

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize)
      if (scatterplot) {
        scatterplot.destroy()
      }
      isInitializedRef.current = false
    }
  }, [xTrainData, yTrainData, categoryData, categoryLabels])

  // Handle test data updates - redraw everything when test data changes
  useEffect(() => {
    if (
      !scatterplotRef.current ||
      !isInitializedRef.current ||
      !xTrainData ||
      !yTrainData ||
      !categoryData
    ) {
      return
    }

    // Only redraw if we have test data, otherwise training data is already rendered
    if (xTestData.length === 0 && yTestData.length === 0) {
      return
    }

    const scatterplot = scatterplotRef.current
    const numPoints = Math.min(xTrainData.length, yTrainData.length, categoryData.length)

    // Get training data
    let xArray: Float32Array
    let yArray: Float32Array

    if (xTrainData.data.length > 0 && xTrainData.data[0].values instanceof Float32Array) {
      xArray = xTrainData.data[0].values as Float32Array
    } else {
      xArray = new Float32Array(numPoints)
      for (let i = 0; i < numPoints; i++) {
        xArray[i] = xTrainData.get(i) || 0
      }
    }

    if (yTrainData.data.length > 0 && yTrainData.data[0].values instanceof Float32Array) {
      yArray = yTrainData.data[0].values as Float32Array
    } else {
      yArray = new Float32Array(numPoints)
      for (let i = 0; i < numPoints; i++) {
        yArray[i] = yTrainData.get(i) || 0
      }
    }

    let categoryArrayData: number[]
    if (categoryData.data.length > 0 && categoryData.data[0].values) {
      categoryArrayData = Array.from(categoryData.data[0].values)
    } else {
      categoryArrayData = Array.from(
        { length: categoryData.length },
        (_, i) => categoryData.get(i) || 0
      )
    }

    // Combine training and test data
    const trainX = Array.from(xArray)
    const trainY = Array.from(yArray)
    const allX = new Float32Array([...trainX, ...xTestData])
    const allY = new Float32Array([...trainY, ...yTestData])

    // Create category data for test points (use a special category for test data)
    const categoryColors = generateCategoryColors(categoryLabels.length + 1)
    const testCategories = new Array(xTestData.length).fill(categoryColors.length - 1)
    const allCategories = [...categoryArrayData, ...testCategories]

    // Create size data: 0 for training data (first pointSize), 1 for test data (second pointSize)
    const trainSize = new Array(numPoints).fill(0)
    const testSize = new Array(xTestData.length).fill(1)
    const allSizes = [...trainSize, ...testSize]

    const columnData = {
      x: allX,
      y: allY,
      valueA: allCategories,
      valueB: allSizes,
    }

    // Redraw all points (training + test)
    const redrawAllData = async () => {
      if (isDrawingRef.current) return
      isDrawingRef.current = true
      try {
        await scatterplot.draw(columnData)
        console.log(
          'Redrew all points:',
          allX.length,
          'total (',
          numPoints,
          'training +',
          xTestData.length,
          'test)'
        )
      } finally {
        isDrawingRef.current = false
      }
    }
    redrawAllData()
  }, [xTestData, yTestData, xTrainData, yTrainData, categoryData, categoryLabels])

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100%',
        overflow: 'hidden', // Prevent any overflow
      }}
    >
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          height: '100%',
          display: 'block',
        }}
      />
    </div>
  )
}

export default ScatterPlotWebGL

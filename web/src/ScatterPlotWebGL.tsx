import React, { useRef, useEffect } from 'react'
import createScatterplot from 'regl-scatterplot'
import { Vector } from 'apache-arrow'

// Generate distinct colors for categories
const generateCategoryColors = (numCategories: number): number[][] => {
  const colors: number[][] = []

  // Use HSL to generate evenly spaced colors
  for (let i = 0; i < numCategories; i++) {
    const hue = (i * 360) / numCategories
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

    colors.push([
      Math.round((r + m) * 255),
      Math.round((g + m) * 255),
      Math.round((b + m) * 255),
      255, // Alpha
    ])
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
  isRunning?: boolean // Add prop to control rendering behavior
  onRef?: (ref: { drawNewPoints: (x: number[], y: number[]) => void } | null) => void
}

const ScatterPlotWebGL: React.FC<ScatterPlotWebGLProps> = ({
  xTrainData,
  yTrainData,
  xTestData,
  yTestData,
  categoryData,
  categoryLabels,
  isRunning = false,
  onRef,
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const scatterplotRef = useRef<ReturnType<typeof createScatterplot> | null>(null)

  useEffect(() => {
    if (!containerRef.current || !canvasRef.current || !xTrainData || !yTrainData || !categoryData)
      return

    // console.log('Setting up ScatterPlotWebGL with categories:', {
    //   numPoints: xData.length,
    //   numCategories: categoryLabels.length,
    //   categoryLabels,
    //   categoryRange: [
    //     Math.min(
    //       ...(categoryData.data[0]?.values ||
    //         Array.from({ length: categoryData.length }, (_, i) => categoryData.get(i) || 0))
    //     ),
    //     Math.max(
    //       ...(categoryData.data[0]?.values ||
    //         Array.from({ length: categoryData.length }, (_, i) => categoryData.get(i) || 0))
    //     ),
    //   ],
    // })

    // Generate colors for each category + one extra color for test points
    const categoryColors = generateCategoryColors(categoryLabels.length + 1)

    // Set the last color to be a distinct color for test points (e.g., bright red)
    categoryColors[categoryColors.length - 1] = [255, 0, 0, 255] // Bright red for test points

    // Create the scatterplot instance
    const scatterplot = createScatterplot({
      canvas: canvasRef.current,
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
      // performanceMode: true, // Enable performance mode for better rendering
    })

    // Set up categorical coloring
    scatterplot.set({
      pointSize: 5.0, // Slightly larger default size
      pointColor: categoryColors,
      colorBy: 'valueA',
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

    // Use column-based data format with valueA for categorical coloring
    // Include both training and existing test data
    const trainX = Array.from(xArray)
    const trainY = Array.from(yArray)

    // For debugging: disable training data when running to see test points clearly
    const showTrainingData = !isRunning

    const allX = new Float32Array(showTrainingData ? [...trainX, ...xTestData] : [...xTestData])
    const allY = new Float32Array(showTrainingData ? [...trainY, ...yTestData] : [...yTestData])

    // Create category data for test points (use a special category for test data)
    const testCategories = new Array(xTestData.length).fill(categoryColors.length - 1)
    const allCategories = showTrainingData
      ? [...categoryArrayData, ...testCategories]
      : testCategories

    const columnData = {
      x: allX,
      y: allY,
      valueA: allCategories,
    }

    // Draw the points using column format
    scatterplot.draw(columnData)

    console.log(
      'Drew',
      allX.length,
      'points (',
      showTrainingData ? numPoints : 0,
      'training +',
      xTestData.length,
      'test) with',
      categoryColors.length,
      'category colors using valueA'
    )

    // Create a reference object with drawNewPoints method
    const scatterplotRefObj = {
      drawNewPoints: (newX: number[], newY: number[]) => {
        // This function is mainly for triggering redraws
        // The actual new data should already be added to xTestData/yTestData by the parent
        // and the useEffect will handle the redraw automatically
        console.log(
          'drawNewPoints called with',
          newX.length,
          'new points at coordinates:',
          newX[0],
          newY[0]
        )
      },
    }

    // Set the reference for parent component
    if (onRef) {
      onRef(scatterplotRefObj)
    }

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
      if (onRef) {
        onRef(null)
      }
    }
  }, [xTrainData, yTrainData, xTestData, yTestData, categoryData, categoryLabels, isRunning, onRef])

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

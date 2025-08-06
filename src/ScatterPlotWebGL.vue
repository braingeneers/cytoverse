<template>
  <div
    ref="containerRef"
    class="scatterplot-container"
  >
    <canvas
      ref="canvasRef"
      class="scatterplot-canvas"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import createScatterplot from 'regl-scatterplot'
import { Vector } from 'apache-arrow'

// Props interface
interface Props {
  xTrainData: Vector | null
  yTrainData: Vector | null
  xTestData: number[]
  yTestData: number[]
  testDataLabels: number[]
  categoryData: Vector | null
  categoryLabels: string[]
}

const props = defineProps<Props>()

// Template refs
const containerRef = ref<HTMLDivElement>()
const canvasRef = ref<HTMLCanvasElement>()

// Component state
let scatterplotRef: ReturnType<typeof createScatterplot> | null = null
const isInitializedRef = ref(false)
const isDrawingRef = ref(false)

// Generate distinct colors for categories + one extra color for test points
// This function generates a set of colors that are evenly spaced in the HSL color space,
// avoiding bright red (hue 0) to ensure good visibility and distinction between categories.
// It returns an array of hex color strings.
// The number of categories is passed as an argument, and it generates colors starting from hue 30 (yellow) to hue 360 (red), ensuring that the colors are visually distinct and not too bright or saturated.
// The colors are generated in HSL format and converted to hex format for use in the scatterplot.
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

  colors.push('#999999') // Add gray for un-labeled query cells

  return colors
}

// Handle resize
const handleResize = () => {
  if (containerRef.value && scatterplotRef) {
    scatterplotRef.set({
      width: containerRef.value.clientWidth,
      height: containerRef.value.clientHeight,
    })
  }
}

// Initial setup and training data rendering
const initializeScatterplot = () => {
  if (!containerRef.value || !canvasRef.value || !props.xTrainData || !props.yTrainData || !props.categoryData)
    return

  // Generate colors for each category + one extra color for test points
  const categoryColors = generateCategoryColors(props.categoryLabels.length)

  // Create the scatterplot instance
  const scatterplot = createScatterplot({
    canvas: canvasRef.value,
    width: containerRef.value.clientWidth,
    height: containerRef.value.clientHeight,
    pointSize: [1, 6, 4], // Use a range for point sizes: training, unlabeled test, labeled test
    // performanceMode: true, // Enable performance mode for better rendering
  })

  // Set up categorical coloring and sizing
  scatterplot.set({
    pointColor: categoryColors,
    colorBy: 'valueA',
    sizeBy: 'valueB',
  })

  scatterplotRef = scatterplot

  // Convert Arrow Vectors to typed arrays for regl-scatterplot
  // Try to access the underlying buffer directly if possible, otherwise convert
  const numPoints = Math.min(props.xTrainData.length, props.yTrainData.length, props.categoryData.length)

  // Check if we can access the underlying typed array directly
  let xArray: Float32Array
  let yArray: Float32Array

  // Arrow Vector.data should give us access to the underlying data
  if (props.xTrainData.data.length > 0 && props.xTrainData.data[0].values instanceof Float32Array) {
    xArray = props.xTrainData.data[0].values as Float32Array
  } else {
    // Fallback to copying data
    xArray = new Float32Array(numPoints)
    for (let i = 0; i < numPoints; i++) {
      xArray[i] = props.xTrainData.get(i) || 0
    }
  }

  if (props.yTrainData.data.length > 0 && props.yTrainData.data[0].values instanceof Float32Array) {
    yArray = props.yTrainData.data[0].values as Float32Array
  } else {
    // Fallback to copying data
    yArray = new Float32Array(numPoints)
    for (let i = 0; i < numPoints; i++) {
      yArray[i] = props.yTrainData.get(i) || 0
    }
  }

  // Get category indices for valueA
  let categoryArrayData: number[]
  if (props.categoryData.data.length > 0 && props.categoryData.data[0].values) {
    categoryArrayData = Array.from(props.categoryData.data[0].values)
  } else {
    // Fallback to vector access
    categoryArrayData = Array.from(
      { length: props.categoryData.length },
      (_, i) => props.categoryData!.get(i) || 0
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
    isInitializedRef.value = true
    console.log(
      'Drew',
      numPoints,
      'training points with',
      categoryColors.length,
      'category colors'
    )
  }
  drawInitialData()
}

// Handle test data updates - redraw everything when test data changes
const updateTestData = () => {
  if (
    !scatterplotRef ||
    !isInitializedRef.value ||
    !props.xTrainData ||
    !props.yTrainData ||
    !props.categoryData
  ) {
    return
  }

  // If test data was cleared, redraw with only training data
  if (props.xTestData.length === 0 && props.yTestData.length === 0) {
    const scatterplot = scatterplotRef
    const numPoints = Math.min(props.xTrainData.length, props.yTrainData.length, props.categoryData.length)

    // Get training data for redraw
    let xArray: Float32Array
    let yArray: Float32Array

    if (props.xTrainData.data.length > 0 && props.xTrainData.data[0].values instanceof Float32Array) {
      xArray = props.xTrainData.data[0].values as Float32Array
    } else {
      xArray = new Float32Array(numPoints)
      for (let i = 0; i < numPoints; i++) {
        xArray[i] = props.xTrainData.get(i) || 0
      }
    }

    if (props.yTrainData.data.length > 0 && props.yTrainData.data[0].values instanceof Float32Array) {
      yArray = props.yTrainData.data[0].values as Float32Array
    } else {
      yArray = new Float32Array(numPoints)
      for (let i = 0; i < numPoints; i++) {
        yArray[i] = props.yTrainData.get(i) || 0
      }
    }

    let categoryArrayData: number[]
    if (props.categoryData.data.length > 0 && props.categoryData.data[0].values) {
      categoryArrayData = Array.from(props.categoryData.data[0].values)
    } else {
      categoryArrayData = Array.from(
        { length: props.categoryData.length },
        (_, i) => props.categoryData!.get(i) || 0
      )
    }

    const trainX = Array.from(xArray)
    const trainY = Array.from(yArray)
    const trainSize = new Array(numPoints).fill(0)

    const columnData = {
      x: new Float32Array(trainX),
      y: new Float32Array(trainY),
      valueA: categoryArrayData,
      valueB: trainSize,
    }

    // Redraw with only training data
    scatterplot.draw(columnData).then(() => {
      console.log('Cleared test data - showing only training points')
    })

    return
  }

  const scatterplot = scatterplotRef
  const numPoints = Math.min(props.xTrainData.length, props.yTrainData.length, props.categoryData.length)

  // Get training data
  let xArray: Float32Array
  let yArray: Float32Array

  if (props.xTrainData.data.length > 0 && props.xTrainData.data[0].values instanceof Float32Array) {
    xArray = props.xTrainData.data[0].values as Float32Array
  } else {
    xArray = new Float32Array(numPoints)
    for (let i = 0; i < numPoints; i++) {
      xArray[i] = props.xTrainData.get(i) || 0
    }
  }

  if (props.yTrainData.data.length > 0 && props.yTrainData.data[0].values instanceof Float32Array) {
    yArray = props.yTrainData.data[0].values as Float32Array
  } else {
    yArray = new Float32Array(numPoints)
    for (let i = 0; i < numPoints; i++) {
      yArray[i] = props.yTrainData.get(i) || 0
    }
  }

  let categoryArrayData: number[]
  if (props.categoryData.data.length > 0 && props.categoryData.data[0].values) {
    categoryArrayData = Array.from(props.categoryData.data[0].values)
  } else {
    categoryArrayData = Array.from(
      { length: props.categoryData.length },
      (_, i) => props.categoryData!.get(i) || 0
    )
  }

  // Combine training and test data
  const trainX = Array.from(xArray)
  const trainY = Array.from(yArray)
  const allX = new Float32Array([...trainX, ...props.xTestData])
  const allY = new Float32Array([...trainY, ...props.yTestData])

  // Create category data for test points
  const categoryColors = generateCategoryColors(props.categoryLabels.length)

  // Use actual labels when available, fallback to gray (last color) for unlabeled points
  const testCategories = props.xTestData.map((_, index) => {
    if (index < props.testDataLabels.length) {
      const labelIndex = props.testDataLabels[index]
      // Use the actual label if valid, otherwise use gray (last color)
      return labelIndex >= 0 && labelIndex < props.categoryLabels.length
        ? labelIndex
        : categoryColors.length - 1
    }
    return categoryColors.length - 1 // Default to gray for points without labels yet
  })

  const allCategories = [...categoryArrayData.slice(0, numPoints), ...testCategories]

  // Create size data: 0 for training data (first pointSize), 1 for unlabeled test data (second pointSize), 2 for labeled test data (third pointSize)
  const trainSize = new Array(numPoints).fill(0)
  const testSize = props.xTestData.map((_, index) => {
    if (index < props.testDataLabels.length) {
      const labelIndex = props.testDataLabels[index]
      // Use size 2 for labeled points (valid label), size 1 for unlabeled
      return labelIndex >= 0 && labelIndex < props.categoryLabels.length ? 2 : 1
    }
    return 1 // Default to unlabeled size for points without labels yet
  })
  const allSizes = [...trainSize, ...testSize]

  const columnData = {
    x: allX,
    y: allY,
    valueA: allCategories,
    valueB: allSizes,
  }

  // Redraw all points (training + test)
  const redrawAllData = async () => {
    if (isDrawingRef.value) {
      console.log(
        '❌ Skipping redraw - already drawing. Current test labels count:',
        props.testDataLabels.filter((l) => l >= 0).length
      )
      return
    }
    isDrawingRef.value = true
    try {
      await scatterplot.draw(columnData)
      console.log(
        'Redrew all points:',
        allX.length,
        'total (',
        numPoints,
        'training +',
        props.xTestData.length,
        'test)'
      )
    } finally {
      isDrawingRef.value = false
    }
  }
  redrawAllData()
}

// Watchers for reactive updates
watch(
  [() => props.xTrainData, () => props.yTrainData, () => props.categoryData, () => props.categoryLabels],
  () => {
    if (scatterplotRef) {
      scatterplotRef.destroy()
      scatterplotRef = null
      isInitializedRef.value = false
    }
    initializeScatterplot()
  }
)

watch(
  [() => props.xTestData, () => props.yTestData, () => props.testDataLabels],
  () => {
    updateTestData()
  },
  { deep: true }
)

onMounted(() => {
  initializeScatterplot()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  if (scatterplotRef) {
    scatterplotRef.destroy()
  }
})
</script>

<style scoped>
.scatterplot-container {
  width: 100%;
  height: 100%;
  overflow: hidden;
}

.scatterplot-canvas {
  width: 100%;
  height: 100%;
  display: block;
}
</style>
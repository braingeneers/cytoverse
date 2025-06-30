declare module 'regl-scatterplot' {
  interface ScatterplotOptions {
    canvas?: HTMLCanvasElement
    width?: number
    height?: number
    backgroundColor?: string | number[]
    pointColor?: string | number[] | number[][]
    pointColorActive?: string | number[]
    pointColorHover?: string | number[]
    pointSize?: number
    opacity?: number
    opacityBy?: string | number[]
    colorBy?: string | number[]
  }

  interface DrawOptions {
    colorBy?: number[]
    sizeBy?: number[]
    opacityBy?: number[]
  }

  interface ColumnData {
    x: number[] | Float32Array
    y: number[] | Float32Array
    valueA?: number[]
    valueB?: number[]
    line?: number[]
    lineOrder?: number[]
  }

  interface ScatterplotInstance {
    draw(points: number[][] | ColumnData, options?: DrawOptions): void
    set(options: Partial<ScatterplotOptions>): void
    destroy(): void
  }

  function createScatterplot(options?: ScatterplotOptions): ScatterplotInstance
  export default createScatterplot
}

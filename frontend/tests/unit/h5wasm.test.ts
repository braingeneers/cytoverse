import { describe, it, expect, beforeAll } from 'vitest'
import h5wasm, { type File as H5File, type Group, type Dataset } from 'h5wasm'
import { resolve, dirname } from 'path'

interface H5Module {
  FS: {
    writeFile: (name: string, data: Uint8Array) => void
    mount: (fs: any, options: any, mountPoint: string) => void
    filesystems: {
      NODEFS?: any
    }
    mkdir: (path: string) => void
    analyzePath: (path: string) => { exists: boolean }
  }
}

describe('h5wasm H5AD File Loading', () => {
  let Module: H5Module

  beforeAll(async () => {
    Module = await h5wasm.ready as H5Module
  })

  it('should open file directly from disk using NODEFS (if available)', async () => {
    const h5FilePath = resolve(__dirname, '../../../fixtures/GSE136831_subsample_10.h5ad')
    const h5Dir = dirname(h5FilePath)
    const h5FileName = 'GSE136831_subsample_10.h5ad'
    
    let file: H5File
    
    // Try to use NODEFS if available
    if (Module.FS.filesystems?.NODEFS) {
      console.log('Using NODEFS to mount file directly from disk')
      
      // Create mount point
      if (!Module.FS.analyzePath('/fixtures').exists) {
        Module.FS.mkdir('/fixtures')
      }
      
      // Mount the fixtures directory
      Module.FS.mount(Module.FS.filesystems.NODEFS, {
        root: h5Dir
      }, '/fixtures')
      
      file = new h5wasm.File(`/fixtures/${h5FileName}`, 'r') as H5File
    } else {
      console.log('NODEFS not available, falling back to in-memory loading')
      // Fallback to loading file into memory
      const { readFileSync } = await import('fs')
      const fileBuffer = readFileSync(h5FilePath)
      Module.FS.writeFile('test.h5ad', new Uint8Array(fileBuffer))
      file = new h5wasm.File('test.h5ad', 'r') as H5File
    }
    
    expect(file).toBeDefined()
    expect(file.keys()).toContain('X')
    expect(file.keys()).toContain('obs')
    expect(file.keys()).toContain('var')
    
    const xEntity = file.get('X')
    expect(xEntity).toBeDefined()
    
    if (!xEntity) {
      throw new Error('X group/dataset not found')
    }
    
    const isGroup = 'keys' in xEntity
    
    if (isGroup) {
      const xGroup = xEntity as Group
      const isSparse = xGroup.keys().includes('data') && 
                      xGroup.keys().includes('indices') && 
                      xGroup.keys().includes('indptr')
      
      expect(isSparse).toBe(true)
      
      if (isSparse) {
        const dataDataset = xGroup.get('data') as Dataset
        expect(dataDataset).toBeDefined()
        expect(dataDataset.shape).toBeDefined()
        expect(dataDataset.shape?.length).toBeGreaterThan(0)
      }
    } else {
      const xDataset = xEntity as Dataset
      expect(xDataset.shape).toBeDefined()
      expect(xDataset.shape?.length).toBe(2)
    }
    
    const obsGroup = file.get('obs')
    expect(obsGroup).toBeDefined()
    
    const varGroup = file.get('var')
    expect(varGroup).toBeDefined()
    
    file.close()
  })

  it('should read dimensions from sparse H5AD file', async () => {
    const h5FilePath = resolve(__dirname, '../../../fixtures/GSE136831_subsample_10.h5ad')
    const h5Dir = dirname(h5FilePath)
    const h5FileName = 'GSE136831_subsample_10.h5ad'
    
    let file: H5File
    
    if (Module.FS.filesystems?.NODEFS && !Module.FS.analyzePath('/fixtures').exists) {
      Module.FS.mkdir('/fixtures')
      Module.FS.mount(Module.FS.filesystems.NODEFS, {
        root: h5Dir
      }, '/fixtures')
    }
    
    if (Module.FS.filesystems?.NODEFS) {
      file = new h5wasm.File(`/fixtures/${h5FileName}`, 'r') as H5File
    } else {
      const { readFileSync } = await import('fs')
      const fileBuffer = readFileSync(h5FilePath)
      Module.FS.writeFile('test_dims.h5ad', new Uint8Array(fileBuffer))
      file = new h5wasm.File('test_dims.h5ad', 'r') as H5File
    }
    
    const xEntity = file.get('X')
    if (!xEntity) {
      throw new Error('X group/dataset not found')
    }
    
    const isGroup = 'keys' in xEntity
    
    let nCells = 0
    let nGenes = 0
    
    if (isGroup) {
      const xGroup = xEntity as Group
      const indptrDataset = xGroup.get('indptr') as Dataset
      const indicesDataset = xGroup.get('indices') as Dataset
      
      if (indptrDataset && indicesDataset && indptrDataset.shape && indicesDataset.value) {
        nCells = indptrDataset.shape[0] - 1
        const indicesMax = Math.max(...Array.from(indicesDataset.value as Float32Array))
        nGenes = indicesMax + 1
      }
    } else {
      const xDataset = xEntity as Dataset;
      if (xDataset.shape) {
        nCells = xDataset.shape[0];
        nGenes = xDataset.shape[1];
      } else {
        throw new Error('xDataset.shape is null');
      }
    }
    
    expect(nCells).toBeGreaterThan(0)
    expect(nGenes).toBeGreaterThan(0)
    
    console.log(`File contains ${nCells} cells and ${nGenes} genes`)
    
    file.close()
  })

  it('should detect sparse vs dense matrix format', async () => {
    const h5FilePath = resolve(__dirname, '../../../fixtures/GSE136831_subsample_10.h5ad')
    const h5Dir = dirname(h5FilePath)
    const h5FileName = 'GSE136831_subsample_10.h5ad'
    
    let file: H5File
    
    if (Module.FS.filesystems?.NODEFS && !Module.FS.analyzePath('/fixtures').exists) {
      Module.FS.mkdir('/fixtures')
      Module.FS.mount(Module.FS.filesystems.NODEFS, {
        root: h5Dir
      }, '/fixtures')
    }
    
    if (Module.FS.filesystems?.NODEFS) {
      file = new h5wasm.File(`/fixtures/${h5FileName}`, 'r') as H5File
    } else {
      const { readFileSync } = await import('fs')
      const fileBuffer = readFileSync(h5FilePath)
      Module.FS.writeFile('test_format.h5ad', new Uint8Array(fileBuffer))
      file = new h5wasm.File('test_format.h5ad', 'r') as H5File
    }
    
    const xEntity = file.get('X')
    if (!xEntity) {
      throw new Error('X group/dataset not found')
    }
    
    const isGroup = 'keys' in xEntity
    const isSparse = isGroup && 
                    (xEntity as Group).keys().includes('data') && 
                    (xEntity as Group).keys().includes('indices') && 
                    (xEntity as Group).keys().includes('indptr')
    
    const isDense = !isGroup && 'dtype' in xEntity
    
    expect(isSparse || isDense).toBe(true)
    
    if (isSparse) {
      console.log('Matrix is in sparse format')
      const xGroup = xEntity as Group
      expect(xGroup.get('data')).toBeDefined()
      expect(xGroup.get('indices')).toBeDefined()
      expect(xGroup.get('indptr')).toBeDefined()
    } else {
      console.log('Matrix is in dense format')
      const xDataset = xEntity as Dataset
      expect(xDataset.dtype).toBeDefined()
    }
    
    file.close()
  })
})
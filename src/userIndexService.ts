import { openDB, DBSchema, IDBPDatabase } from 'idb'

export interface UserIndexPartition {
  pqCodes: Uint8Array  // Concatenated PQ codes
  pqCodeCount: number  // Number of PQ codes (to know how to split)
  labelIndices: Int32Array
  cellIndices: Int32Array
}

export interface UserIndex {
  id: string
  name: string
  baseModelId: string
  created: Date
  cellCount: number
  partitions: Record<number, UserIndexPartition>
  coordinates: {
    x: Float32Array
    y: Float32Array
  }
  labels: string[]
}

interface UserIndexDB extends DBSchema {
  userIndexes: {
    key: string
    value: UserIndex
  }
}

class UserIndexService {
  private db: IDBPDatabase<UserIndexDB> | null = null
  private readonly DB_NAME = 'cytoverse-user-indexes'
  private readonly DB_VERSION = 1

  async init(): Promise<void> {
    this.db = await openDB<UserIndexDB>(this.DB_NAME, this.DB_VERSION, {
      upgrade(db) {
        if (!db.objectStoreNames.contains('userIndexes')) {
          db.createObjectStore('userIndexes', { keyPath: 'id' })
        }
      }
    })
  }

  private ensureDb(): IDBPDatabase<UserIndexDB> {
    if (!this.db) {
      throw new Error('UserIndexService not initialized. Call init() first.')
    }
    return this.db
  }

  async saveUserIndex(index: UserIndex): Promise<void> {
    const db = this.ensureDb()
    
    // Create a clean copy to ensure all data is serializable
    const cleanIndex: UserIndex = {
      id: index.id,
      name: index.name,
      baseModelId: index.baseModelId,
      created: index.created,
      cellCount: index.cellCount,
      partitions: {},
      coordinates: {
        x: index.coordinates.x,
        y: index.coordinates.y
      },
      labels: [...index.labels] // Create a new array copy
    }
    
    // Deep copy partitions to ensure clean structure
    for (const [key, partition] of Object.entries(index.partitions)) {
      cleanIndex.partitions[parseInt(key)] = {
        pqCodes: new Uint8Array(partition.pqCodes),
        pqCodeCount: partition.pqCodeCount,
        labelIndices: new Int32Array(partition.labelIndices),
        cellIndices: new Int32Array(partition.cellIndices)
      }
    }
    
    await db.put('userIndexes', cleanIndex)
  }

  async getUserIndex(id: string): Promise<UserIndex | undefined> {
    const db = this.ensureDb()
    return await db.get('userIndexes', id)
  }

  async getAllUserIndexes(): Promise<UserIndex[]> {
    const db = this.ensureDb()
    return await db.getAll('userIndexes')
  }

  async deleteUserIndex(id: string): Promise<void> {
    const db = this.ensureDb()
    await db.delete('userIndexes', id)
  }

  async getUserIndexMetadata(): Promise<Array<{
    id: string
    name: string
    baseModelId: string
    created: Date
    cellCount: number
  }>> {
    const indexes = await this.getAllUserIndexes()
    return indexes.map(index => ({
      id: index.id,
      name: index.name,
      baseModelId: index.baseModelId,
      created: index.created,
      cellCount: index.cellCount
    }))
  }

  transformArtifactsToUserIndex(
    name: string,
    baseModelId: string,
    artifacts: {
      partitionIds: Int32Array
      pqCodes: Uint8Array[]
      labelIndices: Int32Array
      x: Float32Array
      y: Float32Array
    },
    labels: string[]
  ): UserIndex {
    const partitions: Record<number, UserIndexPartition> = {}
    
    // First pass: collect cells by partition
    const partitionGroups: Record<number, {
      pqCodes: Uint8Array[]
      labelIndices: number[]
      cellIndices: number[]
    }> = {}
    
    artifacts.partitionIds.forEach((partitionId, cellIndex) => {
      if (!partitionGroups[partitionId]) {
        partitionGroups[partitionId] = {
          pqCodes: [],
          labelIndices: [],
          cellIndices: []
        }
      }
      
      partitionGroups[partitionId].pqCodes.push(artifacts.pqCodes[cellIndex])
      partitionGroups[partitionId].labelIndices.push(artifacts.labelIndices[cellIndex])
      partitionGroups[partitionId].cellIndices.push(cellIndex)
    })
    
    // Second pass: create concatenated arrays for each partition
    Object.entries(partitionGroups).forEach(([partitionIdStr, group]) => {
      const partitionId = parseInt(partitionIdStr)
      const pqCodeLength = group.pqCodes[0]?.length || 0
      
      // Concatenate all PQ codes into a single Uint8Array
      const concatenatedPqCodes = new Uint8Array(group.pqCodes.length * pqCodeLength)
      group.pqCodes.forEach((pqCode, i) => {
        concatenatedPqCodes.set(pqCode, i * pqCodeLength)
      })
      
      partitions[partitionId] = {
        pqCodes: concatenatedPqCodes,
        pqCodeCount: group.pqCodes.length,
        labelIndices: new Int32Array(group.labelIndices),
        cellIndices: new Int32Array(group.cellIndices)
      }
    })
    
    return {
      id: `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      name,
      baseModelId,
      created: new Date(),
      cellCount: artifacts.partitionIds.length,
      partitions,
      coordinates: {
        x: artifacts.x,
        y: artifacts.y
      },
      labels
    }
  }

  async close(): Promise<void> {
    if (this.db) {
      this.db.close()
      this.db = null
    }
  }
}

export const userIndexService = new UserIndexService()
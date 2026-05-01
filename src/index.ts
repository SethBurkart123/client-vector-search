import FastPriorityQueue from 'fastpriorityqueue';

const DEFAULT_TOP_K = 3;

interface Filter {
  [key: string]: any;
}

import Cache from './cache';
import { IndexedDbManager, StoreOptions } from './indexedDB';
import { cosineSimilarity } from './utils';
export { ExperimentalHNSWIndex } from './hnsw';

// uncomment if you want to test indexedDB implementation in node env for faster dev cycle
// import { IDBFactory } from 'fake-indexeddb';
// const indexedDB = new IDBFactory();

export interface SearchResult {
  similarity: number;
  object: any;
}

type StorageOptions = 'indexedDB' | 'localStorage' | 'none';

/**
 * Interface for search options in the EmbeddingIndex class.
 * topK: The number of top similar items to return.
 * filter: An optional filter to apply to the objects before searching.
 * useStorage: A flag to indicate whether to use storage options like indexedDB or localStorage.
 */
interface SearchOptions {
  topK?: number;
  filter?: Filter;
  useStorage?: StorageOptions;
  storageOptions?: { indexedDBName: string; indexedDBObjectStoreName: string }; // TODO: generalize it to localStorage as well
  dedupeEntries?: boolean;
  debug?: boolean;
}

const cacheInstance = Cache.getInstance();

let pipe: any;
let currentModel: string;

export { IndexedDbManager, StoreOptions };

export type DeviceOption = 'webgpu' | 'wasm' | 'cpu' | 'auto';

async function detectDevice(): Promise<DeviceOption | undefined> {
  // In a browser with WebGPU available, prefer it.
  if (
    typeof navigator !== 'undefined' &&
    'gpu' in navigator &&
    (navigator as any).gpu !== null
  ) {
    try {
      const adapter = await (navigator as any).gpu.requestAdapter();
      if (adapter) return 'webgpu';
    } catch {
      // WebGPU probe failed; fall through to runtime default.
    }
  }
  // Otherwise let transformers.js pick the runtime-appropriate default
  // (`wasm` in browsers, `cpu` in Node.js).
  return undefined;
}

export const initializeModel = async (
  model: string = 'Xenova/gte-small',
  pipeline_str: string = 'feature-extraction',
  device?: DeviceOption,
): Promise<void> => {
  if (model !== currentModel) {
    const transformersModule = await import('@huggingface/transformers');
    const pipeline = transformersModule.pipeline;
    const resolvedDevice =
      device === 'auto' || device === undefined
        ? await detectDevice()
        : device;
    const pipelineOptions: { device?: DeviceOption } = {};
    if (resolvedDevice) pipelineOptions.device = resolvedDevice;
    pipe = await pipeline(pipeline_str as any, model, pipelineOptions as any);
    currentModel = model;
  }
};

export const getEmbedding = async (
  text: string,
  precision: number = 7,
  options = { pooling: 'mean', normalize: false },
  model = 'Xenova/gte-small',
  device?: DeviceOption,
): Promise<number[]> => {
  const cachedEmbedding = cacheInstance.get(text);
  if (cachedEmbedding) {
    return Promise.resolve(cachedEmbedding);
  }

  if (model !== currentModel) {
    await initializeModel(model, 'feature-extraction', device);
  }

  const output = await pipe(text, options);
  const roundedOutput = Array.from(output.data as number[]).map(
    (value: number) => parseFloat(value.toFixed(precision)),
  );
  cacheInstance.set(text, roundedOutput);
  return Array.from(roundedOutput);
};

export class EmbeddingIndex {
  private objects: Filter[];
  private keys: string[];
  private indexedDBDataCache: any[] | null = null; // Cache for preloaded data
  private preloadedDBName: string | null = null;
  private preloadedObjectStoreName: string | null = null;

  constructor(initialObjects?: Filter[]) {
    // TODO: add support for options while creating index such as  {... indexedDB: true, ...}
    this.objects = [];
    this.keys = [];
    if (initialObjects && initialObjects.length > 0) {
      initialObjects.forEach((obj) => this.validateAndAdd(obj));
      if (initialObjects[0]) {
        this.keys = Object.keys(initialObjects[0]);
      }
    }

    this.objects = [];
    this.keys = [];
    if (initialObjects && initialObjects.length > 0) {
      initialObjects.forEach((obj) => this.validateAndAdd(obj));
      if (initialObjects[0]) {
        this.keys = Object.keys(initialObjects[0]);
      }
    }
    // Ensure cache properties are initialized
    this.indexedDBDataCache = null;
    this.preloadedDBName = null;
    this.preloadedObjectStoreName = null;
  }

  /**
   * Explicitly preloads all data from the specified IndexedDB object store
   * into an in-memory cache for faster subsequent searches.
   * @param DBname - The name of the IndexedDB database.
   * @param objectStoreName - The name of the object store.
   */
  public async preloadIndexedDB(
    DBname: string = 'embeddiaDB',
    objectStoreName: string = 'embeddiaObjectStore',
    opts: StoreOptions = {},
  ): Promise<void> {
    // Clear any existing cache before preloading new data
    this.clearIndexedDBCache();

    try {
      const fetchedObjects = await new Promise<any[]>((resolve, reject) => {
        const request = indexedDB.open(DBname);

        request.onerror = async () => {
          try {
            await IndexedDbManager.create(DBname, objectStoreName, opts);
            resolve([]); // Return empty array for new database
          } catch (createError) {
            reject(new Error('Failed to create database'));
          }
        };

        request.onsuccess = (event) => {
          const db = (event.target as IDBOpenDBRequest).result;
          if (!db.objectStoreNames.contains(objectStoreName)) {
            db.close();
            IndexedDbManager.create(DBname, objectStoreName, opts)
              .then(() => resolve([]))
              .catch((error) => {
                console.error(`Object store creation failed: ${error}`);
                reject(new Error(`Object store creation failed: ${error}`));
              });
            return;
          }
          const transaction = db.transaction([objectStoreName], 'readonly');
          const objectStore = transaction.objectStore(objectStoreName);
          const getAllRequest = objectStore.getAll();

          getAllRequest.onerror = (event) => {
            console.error('Failed to fetch data during preload:', event);
            db.close();
            reject(new Error('Failed to fetch data during preload'));
          };

          getAllRequest.onsuccess = (event) => {
            resolve((event.target as IDBRequest).result as any[]);
          };

          transaction.oncomplete = () => {
            db.close();
          };
          transaction.onerror = (event) => {
            console.error('Preload transaction error:', event);
            db.close();
          };
        };
      });

      // Store fetched data in the cache
      this.indexedDBDataCache = fetchedObjects;
      this.preloadedDBName = DBname;
      this.preloadedObjectStoreName = objectStoreName;
    } catch (error) {
      console.error('Error during IndexedDB preload:', error);
      this.clearIndexedDBCache(); // Clear potentially partial cache on error
    }
  }

  /**
   * Clears the in-memory cache of preloaded IndexedDB data.
   * Call this if you know the underlying IndexedDB data has changed.
   */
  clearIndexedDBCache(): void {
    if (this.indexedDBDataCache) {
      this.indexedDBDataCache = null;
      this.preloadedDBName = null;
      this.preloadedObjectStoreName = null;
    }
  }

  private findVectorIndex(filter: Filter): number {
    return this.objects.findIndex((object) =>
      Object.keys(filter).every((key) => object[key] === filter[key]),
    );
  }

  private validateAndAdd(obj: Filter) {
    if (!Array.isArray(obj.embedding) || obj.embedding.some(isNaN)) {
      throw new Error(
        'Object must have an embedding property of type number[]',
      );
    }
    if (this.keys.length === 0) {
      this.keys = Object.keys(obj);
    } else if (!this.keys.every((key) => key in obj)) {
      throw new Error(
        'Object must have the same properties as the initial objects',
      );
    }
    this.objects.push(obj);
  }

  add(obj: Filter) {
    this.validateAndAdd(obj);
  }

  // Method to update an existing vector in the index
  update(filter: Filter, vector: Filter) {
    const index = this.findVectorIndex(filter);
    if (index === -1) {
      throw new Error('Vector not found');
    }
    if (vector.hasOwnProperty('embedding')) {
      // Validate and add the new vector
      this.validateAndAdd(vector);
    }
    // Replace the old vector with the new one
    this.objects[index] = Object.assign(this.objects[index] as Filter, vector);
  }

  // Method to remove a vector from the index
  remove(filter: Filter) {
    const index = this.findVectorIndex(filter);
    if (index === -1) {
      throw new Error('Vector not found');
    }
    // Remove the vector from the index
    this.objects.splice(index, 1);
  }

  // Method to remove multiple vectors from the index
  removeBatch(filters: Filter[]) {
    filters.forEach((filter) => {
      const index = this.findVectorIndex(filter);
      if (index !== -1) {
        // Remove the vector from the index
        this.objects.splice(index, 1);
      }
    });
  }

  // Method to retrieve a vector from the index
  get(filter: Filter) {
    const vector = this.objects[this.findVectorIndex(filter)];
    return vector || null;
  }

  size(): number {
    // Returns the size of the index
    return this.objects.length;
  }

  clear() {
    this.objects = [];
  }

  async search(
    queryEmbedding: number[],
    options: SearchOptions = {
      topK: 3,
      useStorage: 'none',
      storageOptions: {
        indexedDBName: 'embeddiaDB',
        indexedDBObjectStoreName: 'embeddiaObjectStore',
      },
      dedupeEntries: false,
      debug: false,
    },
  ): Promise<SearchResult[]> {
    const topK = options.topK || DEFAULT_TOP_K;
    const filter = options.filter || {};
    const useStorage = options.useStorage || 'none';

    if (useStorage === 'indexedDB') {
      const DBname = options.storageOptions?.indexedDBName || 'embeddiaDB';
      const objectStoreName =
        options.storageOptions?.indexedDBObjectStoreName ||
        'embeddiaObjectStore';

      if (typeof indexedDB === 'undefined') {
        console.error('IndexedDB is not supported');
        throw new Error('IndexedDB is not supported');
      }

      return await this.loadAndSearchFromIndexedDB(
        DBname,
        objectStoreName,
        queryEmbedding,
        topK,
        options.dedupeEntries || false,
        filter,
        options.debug || false,
      );
    } else {
      return await this.performCompareSearch(
        options.dedupeEntries || false,
        filter,
        this.objects,
        queryEmbedding,
        topK,
      );
    }
  }

  async saveIndex(
    storageType: string,
    options: { DBName: string; objectStoreName: string } = {
      DBName: 'embeddiaDB',
      objectStoreName: 'embeddiaObjectStore',
    },
  ) {
    if (storageType === 'indexedDB') {
      await this.saveToIndexedDB(options.DBName, options.objectStoreName);
    } else {
      throw new Error(
        `Unsupported storage type: ${storageType} \n Supported storage types: "indexedDB"`,
      );
    }
  }

  async saveToIndexedDB(
    DBname: string = 'embeddiaDB',
    objectStoreName: string = 'embeddiaObjectStore',
    clearFirst: boolean = true,
  ): Promise<void> {
    if (typeof indexedDB === 'undefined') {
      console.error('IndexedDB is not defined');
      throw new Error('IndexedDB is not supported');
    }

    if (!this.objects || this.objects.length === 0) {
      throw new Error('Index is empty. Nothing to save');
    }

    try {
      const db = await IndexedDbManager.create(DBname, objectStoreName);
      if (clearFirst) {
        await db.clearObjectStore(DBname, objectStoreName);
      }
      await db.addToIndexedDB(this.objects);
    } catch (error) {
      console.error('Error saving index to database:', error);
      throw new Error('Error saving index to database');
    }
  }

  async loadAndSearchFromIndexedDB(
    DBname: string = 'embeddiaDB',
    objectStoreName: string = 'embeddiaObjectStore',
    queryEmbedding: number[],
    topK: number,
    dedupeEntries: boolean,
    filter: Filter,
    debug: boolean = false,
  ): Promise<SearchResult[]> {
    let objectsToSearch: any[] = [];

    const isCacheValid =
      this.indexedDBDataCache &&
      this.preloadedDBName === DBname &&
      this.preloadedObjectStoreName === objectStoreName;

    if (!isCacheValid) {
      if (debug) {
        console.log(
          `Cache invalid or missing for ${DBname}/${objectStoreName}. Preloading...`,
        );
      }
      try {
        // Call the existing preload function. It handles fetching AND updating the cache variables.
        await this.preloadIndexedDB(DBname, objectStoreName);
        // If preload succeeded, the cache should now be valid and populated.
        if (!this.indexedDBDataCache) {
          // This should ideally not happen if preloadIndexedDB works correctly, but safeguard anyway.
          throw new Error('Preload completed but cache is still null.');
        }
        objectsToSearch = this.indexedDBDataCache;
      } catch (error) {
        console.error(
          `Failed to preload data for search from ${DBname}/${objectStoreName}:`,
          error,
        );
        return []; // Return empty results if preload fails
      }
    } else {
      if (debug) {
        console.log(`Using preloaded cache for ${DBname}/${objectStoreName}.`);
      }
      // Cache is valid, use it directly
      objectsToSearch = this.indexedDBDataCache!; // Use non-null assertion as isCacheValid checked it
    }

    const results = await this.performCompareSearch(
      dedupeEntries,
      filter,
      objectsToSearch,
      queryEmbedding,
      topK,
    );

    return results;
  }

  async performCompareSearch(
    dedupeEntries: boolean,
    filter: Filter,
    objectsToSearch: any[],
    queryEmbedding: number[],
    topK: number,
  ): Promise<SearchResult[]> {
    const queue = new FastPriorityQueue<SearchResult>(
      (a, b) => a.similarity < b.similarity,
    );

    // Perform search on the determined objectsToSearch (either from cache or fresh preload)
    if (dedupeEntries) {
      const seenMap: Map<string, SearchResult> = new Map();

      for (const record of objectsToSearch) {
        const passesFilter = Object.keys(filter).every(
          (key) => record[key] === filter[key],
        );

        if (!passesFilter) continue;

        const embedding = record?.embedding;
        if (!Array.isArray(embedding)) {
          console.warn('Record missing or has invalid embedding:', record);
          continue;
        }

        const similarity = cosineSimilarity(queryEmbedding, embedding);
        const id = record.id ?? JSON.stringify(record); // Use 'id' or fallback to a stringified version of the record

        const existing = seenMap.get(id);
        if (!existing || similarity > existing.similarity) {
          seenMap.set(id, { similarity, object: record });
        }
      }

      // Now that we have the best one per ID, put the topK into the priority queue
      for (const result of seenMap.values()) {
        if (queue.size < topK) {
          queue.add(result);
        } else {
          const peeked = queue.peek();
          if (peeked && result.similarity > peeked.similarity) {
            queue.poll();
            queue.add(result);
          }
        }
      }
    } else {
      for (const record of objectsToSearch) {
        // 1. Filter Check
        const passesFilter = Object.keys(filter).every(
          (key) => record[key] === filter[key],
        );

        if (passesFilter) {
          // Ensure record.embedding exists and is an array
          const embedding = record?.embedding;
          if (!Array.isArray(embedding)) {
            console.warn('Record missing or has invalid embedding:', record);
            continue; // Skip this record
          }
          const similarity = cosineSimilarity(queryEmbedding, embedding);

          // 3. Queue Operations
          if (queue.size < topK) {
            queue.add({ similarity, object: record });
          } else {
            const peeked = queue.peek();
            if (peeked && similarity > peeked.similarity) {
              queue.poll();
              queue.add({ similarity, object: record });
            }
          }
        }
      }
    }
    const results: SearchResult[] = [];
    while (!queue.isEmpty()) {
      results.push(queue.poll()!);
    }

    return results.reverse();
  }

  async deleteIndexedDB(DBname: string = 'embeddiaDB'): Promise<void> {
    if (typeof indexedDB === 'undefined') {
      console.error('IndexedDB is not defined');
      throw new Error('IndexedDB is not supported');
    }
    return new Promise((resolve, reject) => {
      const request = indexedDB.deleteDatabase(DBname);

      request.onsuccess = () => {
        resolve();
      };
      request.onerror = (event) => {
        console.error('Failed to delete database', event);
        reject(new Error('Failed to delete database'));
      };
    });
  }

  async deleteIndexedDBObjectStore(
    DBname: string = 'embeddiaDB',
    objectStoreName: string = 'embeddiaObjectStore',
  ): Promise<void> {
    const db = await IndexedDbManager.create(DBname, objectStoreName);

    try {
      await db.deleteIndexedDBObjectStoreFromDB(DBname, objectStoreName);
    } catch (error) {
      console.error('Error deleting object store:', error);
      throw new Error('Error deleting object store');
    }
  }

  async getAllObjectsFromIndexedDB(
    DBname: string = 'embeddiaDB',
    objectStoreName: string = 'embeddiaObjectStore',
    opts: StoreOptions = {},
  ): Promise<any[]> {
    const db = await IndexedDbManager.create(DBname, objectStoreName, opts);
    const objects: any[] = [];
    for await (const record of db.dbGenerator()) {
      objects.push(record);
    }
    return objects;
  }
}

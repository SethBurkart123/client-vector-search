import FastPriorityQueue from 'fastpriorityqueue';

const DEFAULT_TOP_K = 3;

interface Filter {
  [key: string]: any;
}

import Cache from './cache';
import { IndexedDbManager } from './indexedDB';
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
}

const cacheInstance = Cache.getInstance();

let pipe: any;
let currentModel: string;

export const initializeModel = async (
  model: string = 'Xenova/gte-small',
): Promise<void> => {
  if (model !== currentModel) {
    const transformersModule = await import('@xenova/transformers');
    const pipeline = transformersModule.pipeline;
    pipe = await pipeline('feature-extraction', model);
    currentModel = model;
  }
};

export const getEmbedding = async (
  text: string,
  precision: number = 7,
  options = { pooling: 'mean', normalize: false },
  model = 'Xenova/gte-small',
): Promise<number[]> => {
  const cachedEmbedding = cacheInstance.get(text);
  if (cachedEmbedding) {
    return Promise.resolve(cachedEmbedding);
  }

  if (model !== currentModel) {
    await initializeModel(model);
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
    DBname: string = 'clientVectorDB',
    objectStoreName: string = 'ClientEmbeddingStore',
  ): Promise<void> {
    console.log(`Preloading data from ${DBname}/${objectStoreName}...`);
    const preloadStartTime = performance.now();

    // Clear any existing cache before preloading new data
    this.clearIndexedDBCache();

    try {
      const fetchedObjects = await new Promise<any[]>((resolve, reject) => {
        const request = indexedDB.open(DBname);

        request.onerror = (event) => {
          console.error('Failed to open database for preloading:', event);
          reject(new Error('Failed to open database for preloading'));
        };

        request.onsuccess = (event) => {
          const db = (event.target as IDBOpenDBRequest).result;
          if (!db.objectStoreNames.contains(objectStoreName)) {
            db.close();
            console.error(
              `Object store '${objectStoreName}' not found during preload.`,
            );
            reject(new Error(`Object store '${objectStoreName}' not found.`));
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

      const preloadEndTime = performance.now();
      console.log(
        `%cSuccessfully preloaded ${
          fetchedObjects.length
        } objects from ${DBname}/${objectStoreName} in ${(
          preloadEndTime - preloadStartTime
        ).toFixed(2)}ms.`,
        'color: green;',
      );
    } catch (error) {
      console.error('Error during IndexedDB preload:', error);
      this.clearIndexedDBCache(); // Clear potentially partial cache on error
      // Optionally re-throw or handle as needed
      // throw error;
    }
  }

  /**
   * Clears the in-memory cache of preloaded IndexedDB data.
   * Call this if you know the underlying IndexedDB data has changed.
   */
  clearIndexedDBCache(): void {
    if (this.indexedDBDataCache) {
      console.log(
        `Clearing preloaded cache for ${this.preloadedDBName}/${this.preloadedObjectStoreName}.`,
      );
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
        indexedDBName: 'clientVectorDB',
        indexedDBObjectStoreName: 'ClientEmbeddingStore',
      },
    },
  ): Promise<SearchResult[]> {
    const topK = options.topK || DEFAULT_TOP_K;
    const filter = options.filter || {};
    const useStorage = options.useStorage || 'none';

    if (useStorage === 'indexedDB') {
      const DBname = options.storageOptions?.indexedDBName || 'clientVectorDB';
      const objectStoreName =
        options.storageOptions?.indexedDBObjectStoreName ||
        'ClientEmbeddingStore';

      if (typeof indexedDB === 'undefined') {
        console.error('IndexedDB is not supported');
        throw new Error('IndexedDB is not supported');
      }

      return await this.loadAndSearchFromIndexedDB(
        DBname,
        objectStoreName,
        queryEmbedding,
        topK,
        filter,
      );
    } else {
      const queue = new FastPriorityQueue<SearchResult>(
        (a, b) => a.similarity < b.similarity,
      );

      for (const obj of this.objects) {
        if (Object.keys(filter).every((key) => obj[key] === filter[key])) {
          const similarity = cosineSimilarity(queryEmbedding, obj.embedding);

          if (queue.size < topK) {
            queue.add({ similarity, object: obj });
          } else if (queue.peek() && similarity > queue.peek()!.similarity) {
            queue.poll(); // Remove lowest
            queue.add({ similarity, object: obj });
          }
        }
      }

      const results: SearchResult[] = [];
      while (!queue.isEmpty()) {
        results.push(queue.poll() as SearchResult);
      }
      results.reverse(); // Since it's a min-heap
      return results;
    }
  }

  printIndex() {
    console.log('Index Content:');
    this.objects.forEach((obj, idx) => {
      console.log(`Item ${idx + 1}:`, obj);
    });
  }

  async saveIndex(
    storageType: string,
    options: { DBName: string; objectStoreName: string } = {
      DBName: 'clientVectorDB',
      objectStoreName: 'ClientEmbeddingStore',
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
    DBname: string = 'clientVectorDB',
    objectStoreName: string = 'ClientEmbeddingStore',
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
      await db.addToIndexedDB(this.objects);
      console.log(
        `Index saved to database '${DBname}' object store '${objectStoreName}'`,
      );
    } catch (error) {
      console.error('Error saving index to database:', error);
      throw new Error('Error saving index to database');
    }
  }

  async loadAndSearchFromIndexedDB(
    DBname: string = 'clientVectorDB',
    objectStoreName: string = 'ClientEmbeddingStore',
    queryEmbedding: number[],
    topK: number,
    filter: Filter,
  ): Promise<SearchResult[]> {
    const functionStartTime = performance.now();
    let objectsToSearch: any[] = [];

    const isCacheValid =
      this.indexedDBDataCache &&
      this.preloadedDBName === DBname &&
      this.preloadedObjectStoreName === objectStoreName;

    if (!isCacheValid) {
      console.log(
        `Cache invalid or missing for ${DBname}/${objectStoreName}. Preloading...`,
      );
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
      console.log(`Using preloaded cache for ${DBname}/${objectStoreName}.`);
      // Cache is valid, use it directly
      objectsToSearch = this.indexedDBDataCache!; // Use non-null assertion as isCacheValid checked it
    }

    const queue = new FastPriorityQueue<SearchResult>(
      (a, b) => a.similarity < b.similarity,
    );

    // Perform search on the determined objectsToSearch (either from cache or fresh preload)
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
    const results: SearchResult[] = [];
    while (!queue.isEmpty()) {
      results.push(queue.poll()!);
    }

    const totalFunctionTime = performance.now() - functionStartTime;
    console.log(
      `%cTotal Search Function Time: ${totalFunctionTime.toFixed(2)}ms`,
      'color: green;',
    );

    return results.reverse();
  }

  async deleteIndexedDB(DBname: string = 'clientVectorDB'): Promise<void> {
    if (typeof indexedDB === 'undefined') {
      console.error('IndexedDB is not defined');
      throw new Error('IndexedDB is not supported');
    }
    return new Promise((resolve, reject) => {
      const request = indexedDB.deleteDatabase(DBname);

      request.onsuccess = () => {
        console.log(`Database '${DBname}' deleted`);
        resolve();
      };
      request.onerror = (event) => {
        console.error('Failed to delete database', event);
        reject(new Error('Failed to delete database'));
      };
    });
  }

  async deleteIndexedDBObjectStore(
    DBname: string = 'clientVectorDB',
    objectStoreName: string = 'ClientEmbeddingStore',
  ): Promise<void> {
    const db = await IndexedDbManager.create(DBname, objectStoreName);

    try {
      await db.deleteIndexedDBObjectStoreFromDB(DBname, objectStoreName);
      console.log(
        `Object store '${objectStoreName}' deleted from database '${DBname}'`,
      );
    } catch (error) {
      console.error('Error deleting object store:', error);
      throw new Error('Error deleting object store');
    }
  }

  async getAllObjectsFromIndexedDB(
    DBname: string = 'clientVectorDB',
    objectStoreName: string = 'ClientEmbeddingStore',
  ): Promise<any[]> {
    const db = await IndexedDbManager.create(DBname, objectStoreName);
    const objects: any[] = [];
    for await (const record of db.dbGenerator()) {
      objects.push(record);
    }
    return objects;
  }
}

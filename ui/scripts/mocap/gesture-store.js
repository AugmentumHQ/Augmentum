// ui/scripts/mocap/gesture-store.js
// IndexedDB CRUD for recorded gesture library

const DB_NAME = 'avatar-testbench';
const DB_VERSION = 3;
const GESTURE_STORE = 'gestures';

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains('glb_files')) {
        db.createObjectStore('glb_files', { keyPath: 'name' });
      }
      if (!db.objectStoreNames.contains('vrm_file')) {
        db.createObjectStore('vrm_file', { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains(GESTURE_STORE)) {
        db.createObjectStore(GESTURE_STORE, { keyPath: 'name' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function saveGesture(gesture) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(GESTURE_STORE, 'readwrite');
    tx.objectStore(GESTURE_STORE).put(gesture);
    tx.oncomplete = () => { db.close(); resolve(); };
    tx.onerror = () => { db.close(); reject(tx.error); };
  });
}

export async function getGesture(name) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(GESTURE_STORE, 'readonly');
    const req = tx.objectStore(GESTURE_STORE).get(name);
    req.onsuccess = () => { db.close(); resolve(req.result || null); };
    req.onerror = () => { db.close(); reject(req.error); };
  });
}

export async function getAllGestures() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(GESTURE_STORE, 'readonly');
    const req = tx.objectStore(GESTURE_STORE).getAll();
    req.onsuccess = () => { db.close(); resolve(req.result); };
    req.onerror = () => { db.close(); reject(req.error); };
  });
}

export async function deleteGesture(name) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(GESTURE_STORE, 'readwrite');
    tx.objectStore(GESTURE_STORE).delete(name);
    tx.oncomplete = () => { db.close(); resolve(); };
    tx.onerror = () => { db.close(); reject(tx.error); };
  });
}

export async function exportAllGestures() {
  const gestures = await getAllGestures();
  return gestures.map(g => ({
    name: g.name,
    duration: g.duration,
    keyframes: g.keyframes,
    createdAt: g.createdAt,
  }));
}

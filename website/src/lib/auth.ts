import {
  signInWithPopup,
  signOut as firebaseSignOut,
  onAuthStateChanged,
  type User,
} from "firebase/auth";
import {
  doc,
  getDoc,
  setDoc,
  collection,
  addDoc,
  deleteDoc,
  getDocs,
  query,
  orderBy,
  limit,
  serverTimestamp,
} from "firebase/firestore";
import { auth, googleProvider, db } from "./firebase";

// ── Types ──────────────────────────────────────────────────────────────
export interface UserPreferences {
  zoneAlertEnabled: boolean;
  bigMoveAlertEnabled: boolean;
  alertEmail: string;
  maxPE: number;
  maxPB: number;
  minMarketCap: number;
  psuOnly: boolean;
  darkMode: boolean;
}

export interface WatchlistItem {
  id?: string;
  symbol: string;
  name: string;
  addedAt: unknown;
}

export interface AlertRecord {
  id?: string;
  type: string;
  message: string;
  triggeredAt: unknown;
}

const DEFAULT_PREFS: UserPreferences = {
  zoneAlertEnabled: true,
  bigMoveAlertEnabled: false,
  alertEmail: "",
  maxPE: 20,
  maxPB: 3,
  minMarketCap: 500,
  psuOnly: false,
  darkMode: true,
};

// ── Auth ───────────────────────────────────────────────────────────────
export async function signInWithGoogle(): Promise<User> {
  const result = await signInWithPopup(auth, googleProvider);
  // Create or update user document on first login
  const user = result.user;
  const userRef = doc(db, "users", user.uid);
  const snap = await getDoc(userRef);
  if (!snap.exists()) {
    await setDoc(userRef, {
      displayName: user.displayName ?? "",
      email: user.email ?? "",
      photoURL: user.photoURL ?? "",
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp(),
      preferences: DEFAULT_PREFS,
    });
  } else {
    // Update last login & profile info
    await setDoc(
      userRef,
      {
        displayName: user.displayName ?? "",
        email: user.email ?? "",
        photoURL: user.photoURL ?? "",
        updatedAt: serverTimestamp(),
      },
      { merge: true }
    );
  }
  return user;
}

export async function signOutUser(): Promise<void> {
  await firebaseSignOut(auth);
}

export function onAuthChange(callback: (user: User | null) => void): () => void {
  return onAuthStateChanged(auth, callback);
}

// ── Preferences ──────────────────────────────────────────────────────
export async function getUserPreferences(uid: string): Promise<UserPreferences> {
  try {
    const snap = await getDoc(doc(db, "users", uid));
    if (snap.exists()) {
      const data = snap.data();
      return { ...DEFAULT_PREFS, ...(data.preferences as Partial<UserPreferences>) };
    }
  } catch (err) {
    console.error("Error fetching preferences:", err);
  }
  return { ...DEFAULT_PREFS };
}

export async function saveUserPreferences(uid: string, prefs: Partial<UserPreferences>): Promise<void> {
  const current = await getUserPreferences(uid);
  const merged = { ...current, ...prefs };
  await setDoc(
    doc(db, "users", uid),
    { preferences: merged, updatedAt: serverTimestamp() },
    { merge: true }
  );
}

// ── Watchlist ─────────────────────────────────────────────────────────
export async function getUserWatchlist(uid: string): Promise<WatchlistItem[]> {
  try {
    const q = query(
      collection(db, "users", uid, "watchlist"),
      orderBy("addedAt", "desc")
    );
    const snap = await getDocs(q);
    return snap.docs.map((d) => ({ id: d.id, ...d.data() })) as WatchlistItem[];
  } catch (err) {
    console.error("Error fetching watchlist:", err);
    return [];
  }
}

export async function addToWatchlist(uid: string, symbol: string, name: string): Promise<string | null> {
  try {
    const ref = await addDoc(collection(db, "users", uid, "watchlist"), {
      symbol,
      name,
      addedAt: serverTimestamp(),
    });
    return ref.id;
  } catch (err) {
    console.error("Error adding to watchlist:", err);
    return null;
  }
}

export async function removeFromWatchlist(uid: string, itemId: string): Promise<void> {
  try {
    await deleteDoc(doc(db, "users", uid, "watchlist", itemId));
  } catch (err) {
    console.error("Error removing from watchlist:", err);
  }
}

// ── Alert History ─────────────────────────────────────────────────────
export async function getAlertHistory(uid: string, maxItems = 20): Promise<AlertRecord[]> {
  try {
    const q = query(
      collection(db, "users", uid, "alerts"),
      orderBy("triggeredAt", "desc"),
      limit(maxItems)
    );
    const snap = await getDocs(q);
    return snap.docs.map((d) => ({ id: d.id, ...d.data() })) as AlertRecord[];
  } catch (err) {
    console.error("Error fetching alert history:", err);
    return [];
  }
}

'use client';

import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';

export interface UserProfile {
  id: string;
  username: string;
  email: string;
  full_name: string;
  role: string;
  created_at?: string;
}

interface AuthContextType {
  user: UserProfile | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  isAuthModalOpen: boolean;
  authModalMode: 'signin' | 'signup';
  openAuthModal: (mode?: 'signin' | 'signup') => void;
  closeAuthModal: () => void;
  loginWithGoogle: (credential: string) => Promise<{ success: boolean; error?: string }>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const TOKEN_STORAGE_KEY = 'jru_auth_token';
const USER_STORAGE_KEY = 'jru_auth_user';

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState<boolean>(false);
  const [authModalMode, setAuthModalMode] = useState<'signin' | 'signup'>('signin');

  // Initialize and validate auth session on browser mount via httpOnly cookie
  useEffect(() => {
    try {
      const storedUser = localStorage.getItem(USER_STORAGE_KEY);
      if (!storedUser) {
        // Never signed in on this browser: nothing to verify, and the check would only return 401.
        setIsLoading(false);
        return;
      }
      setUser(JSON.parse(storedUser));

      // Background session verification via httpOnly cookie
      fetch('/api/v1/auth/me', {
        credentials: 'include',
      })
        .then((res) => {
          if (res.ok) return res.json();
          throw new Error('Session expired');
        })
        .then((verifiedUser) => {
          setUser(verifiedUser);
          localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(verifiedUser));
          localStorage.removeItem(TOKEN_STORAGE_KEY); // purge any legacy insecure token
        })
        .catch(() => {
          // Token invalidated or expired
          localStorage.removeItem(TOKEN_STORAGE_KEY);
          localStorage.removeItem(USER_STORAGE_KEY);
          setToken(null);
          setUser(null);
        })
        .finally(() => {
          setIsLoading(false);
        });
    } catch (e) {
      console.warn('Auth initialization error:', e);
      setIsLoading(false);
    }
  }, []);

  // Synchronize authentication state across browser tabs
  useEffect(() => {
    const handleStorage = (e: StorageEvent) => {
      if (e.key === USER_STORAGE_KEY) {
        if (e.newValue) {
          try {
            setUser(JSON.parse(e.newValue));
          } catch {
            setUser(null);
          }
        } else {
          setUser(null);
          setToken(null);
        }
      } else if (e.key === TOKEN_STORAGE_KEY && !e.newValue) {
        setToken(null);
      } else if (e.key === null) {
        setUser(null);
        setToken(null);
      }
    };
    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
  }, []);

  const openAuthModal = useCallback((mode: 'signin' | 'signup' = 'signin') => {
    setAuthModalMode(mode);
    setIsAuthModalOpen(true);
  }, []);

  const closeAuthModal = useCallback(() => {
    setIsAuthModalOpen(false);
  }, []);

  const loginWithGoogle = useCallback(async (credential: string): Promise<{ success: boolean; error?: string }> => {
    try {
      const res = await fetch('/api/v1/auth/google', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ credential }),
      });

      const data = await res.json();
      if (!res.ok) {
        return { success: false, error: data.detail || 'Failed to authenticate with Google' };
      }

      setToken(data.access_token);
      setUser(data.user);
      localStorage.removeItem('chatbot_session_id');
      localStorage.removeItem(TOKEN_STORAGE_KEY); // Token stored only in httpOnly secure cookie
      localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(data.user));
      setIsAuthModalOpen(false);
      if (typeof window !== 'undefined') window.dispatchEvent(new Event('jru-auth-change'));
      return { success: true };
    } catch (err: any) {
      return { success: false, error: err.message || 'Network error connecting to auth server' };
    }
  }, []);

  const logout = useCallback(async () => {
    try {
      await fetch('/api/v1/auth/logout', { method: 'POST', credentials: 'include' });
    } catch {
      // Offline fallback: server logout notification failed
    } finally {
      localStorage.removeItem(TOKEN_STORAGE_KEY);
      localStorage.removeItem(USER_STORAGE_KEY);
      localStorage.removeItem('chatbot_session_id');
      try {
        const keysToRemove: string[] = [];
        for (let i = 0; i < localStorage.length; i++) {
          const key = localStorage.key(i);
          if (
            key &&
            (key.startsWith('jru_session_variations_') ||
              key.startsWith('chat_intent_') ||
              key === 'jru_pinned_sessions_guest' ||
              key === 'jru_pinned_sessions')
          ) {
            keysToRemove.push(key);
          }
        }
        keysToRemove.forEach((k) => localStorage.removeItem(k));
      } catch {}
      setToken(null);
      setUser(null);
      if (typeof window !== 'undefined') window.dispatchEvent(new Event('jru-auth-change'));
    }
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!user,
        isLoading,
        isAuthModalOpen,
        authModalMode,
        openAuthModal,
        closeAuthModal,
        loginWithGoogle,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

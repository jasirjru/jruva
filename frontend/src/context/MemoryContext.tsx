'use client';

import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { useAuth } from './AuthContext';
import { UserMemory } from '@/types/memory';

interface MemoryContextType {
  memories: UserMemory[];
  isLoading: boolean;
  isMemoryModalOpen: boolean;
  openMemoryModal: () => void;
  closeMemoryModal: () => void;
  addMemory: (fact: string, category?: string) => Promise<{ success: boolean; error?: string }>;
  deleteMemory: (id: string) => Promise<{ success: boolean; error?: string }>;
  clearAllMemories: () => Promise<{ success: boolean; error?: string }>;
  refreshMemories: () => Promise<void>;
}

const MemoryContext = createContext<MemoryContextType | undefined>(undefined);

export const MemoryProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const { token, isAuthenticated } = useAuth();
  const [memories, setMemories] = useState<UserMemory[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isMemoryModalOpen, setIsMemoryModalOpen] = useState<boolean>(false);

  const openMemoryModal = useCallback(() => setIsMemoryModalOpen(true), []);
  const closeMemoryModal = useCallback(() => setIsMemoryModalOpen(false), []);

  const refreshMemories = useCallback(async () => {
    if (!isAuthenticated) {
      setMemories([]);
      return;
    }
    setIsLoading(true);
    try {
      const headers: Record<string, string> = {};
      if (token) headers['Authorization'] = `Bearer ${token}`;
      const res = await fetch('/api/v1/memory', {
        headers,
        credentials: 'include',
      });
      if (res.ok) {
        const data = await res.json();
        if (data && Array.isArray(data.memories)) {
          setMemories(data.memories);
        }
      }
    } catch (err) {
      console.warn('Failed to fetch user memories:', err);
    } finally {
      setIsLoading(false);
    }
  }, [token, isAuthenticated]);

  useEffect(() => {
    refreshMemories();
  }, [refreshMemories]);

  const addMemory = useCallback(
    async (fact: string, category: string = 'general'): Promise<{ success: boolean; error?: string }> => {
      if (!isAuthenticated) return { success: false, error: 'Sign in required to save memories' };
      try {
        const headers: Record<string, string> = { 'Content-Type': 'application/json' };
        if (token) headers['Authorization'] = `Bearer ${token}`;
        const res = await fetch('/api/v1/memory', {
          method: 'POST',
          headers,
          credentials: 'include',
          body: JSON.stringify({ fact, category }),
        });
        const data = await res.json();
        if (!res.ok) {
          return { success: false, error: data.detail || 'Failed to save memory' };
        }
        setMemories((prev) => [data, ...prev.filter((m) => m.id !== data.id)]);
        return { success: true };
      } catch (err: any) {
        return { success: false, error: err.message || 'Network error saving memory' };
      }
    },
    [token, isAuthenticated]
  );

  const deleteMemory = useCallback(
    async (id: string): Promise<{ success: boolean; error?: string }> => {
      if (!isAuthenticated) return { success: false, error: 'Sign in required' };
      try {
        const headers: Record<string, string> = {};
        if (token) headers['Authorization'] = `Bearer ${token}`;
        const res = await fetch(`/api/v1/memory/${id}`, {
          method: 'DELETE',
          headers,
          credentials: 'include',
        });
        if (!res.ok) {
          const data = await res.json();
          return { success: false, error: data.detail || 'Failed to delete memory' };
        }
        setMemories((prev) => prev.filter((m) => m.id !== id));
        return { success: true };
      } catch (err: any) {
        return { success: false, error: err.message || 'Network error deleting memory' };
      }
    },
    [token, isAuthenticated]
  );

  const clearAllMemories = useCallback(async (): Promise<{ success: boolean; error?: string }> => {
    if (!isAuthenticated) return { success: false, error: 'Sign in required' };
    try {
      const headers: Record<string, string> = {};
      if (token) headers['Authorization'] = `Bearer ${token}`;
      const res = await fetch('/api/v1/memory', {
        method: 'DELETE',
        headers,
        credentials: 'include',
      });
      if (!res.ok) {
        const data = await res.json();
        return { success: false, error: data.detail || 'Failed to clear memories' };
      }
      setMemories([]);
      return { success: true };
    } catch (err: any) {
      return { success: false, error: err.message || 'Network error clearing memories' };
    }
  }, [token, isAuthenticated]);

  return (
    <MemoryContext.Provider
      value={{
        memories,
        isLoading,
        isMemoryModalOpen,
        openMemoryModal,
        closeMemoryModal,
        addMemory,
        deleteMemory,
        clearAllMemories,
        refreshMemories,
      }}
    >
      {children}
    </MemoryContext.Provider>
  );
};

export const useMemory = (): MemoryContextType => {
  const context = useContext(MemoryContext);
  if (!context) {
    throw new Error('useMemory must be used within a MemoryProvider');
  }
  return context;
};

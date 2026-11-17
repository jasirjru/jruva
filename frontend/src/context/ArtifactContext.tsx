'use client';

import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';

export type ArtifactType = 'html' | 'svg' | 'react' | 'mermaid' | 'code';

export interface ArtifactVersion {
  version: number;
  content: string;
  timestamp: Date;
  summary?: string;
}

export interface ArtifactItem {
  id: string;
  title: string;
  type: ArtifactType;
  language: string;
  content: string;
  versions?: ArtifactVersion[];
  currentVersionIndex?: number;
}

interface ArtifactContextType {
  activeArtifact: ArtifactItem | null;
  openArtifact: (artifact: ArtifactItem) => void;
  closeArtifact: () => void;
  updateArtifactContent: (content: string) => void;
  setVersion: (versionIndex: number) => void;
  restoreVersion: (versionIndex: number) => void;
}

const ArtifactContext = createContext<ArtifactContextType | undefined>(undefined);

export const ArtifactProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [activeArtifact, setActiveArtifact] = useState<ArtifactItem | null>(null);

  const openArtifact = useCallback((artifact: ArtifactItem) => {
    setActiveArtifact((prev) => {
      // If reopening or updating same artifact with new content, push new version
      if (prev && (prev.id === artifact.id || prev.title === artifact.title)) {
        if (prev.content !== artifact.content) {
          const existingVersions = prev.versions || [
            { version: 1, content: prev.content, timestamp: new Date() },
          ];
          const newVerNum = existingVersions.length + 1;
          const updatedVersions: ArtifactVersion[] = [
            ...existingVersions,
            { version: newVerNum, content: artifact.content, timestamp: new Date() },
          ];
          return {
            ...artifact,
            content: artifact.content,
            versions: updatedVersions,
            currentVersionIndex: updatedVersions.length - 1,
          };
        }
        return prev;
      }

      // New artifact: initialize version 1
      const initialVersions: ArtifactVersion[] = artifact.versions || [
        { version: 1, content: artifact.content, timestamp: new Date() },
      ];
      return {
        ...artifact,
        versions: initialVersions,
        currentVersionIndex: initialVersions.length - 1,
      };
    });
  }, []);

  const closeArtifact = useCallback(() => {
    setActiveArtifact(null);
  }, []);

  const updateArtifactContent = useCallback((content: string) => {
    setActiveArtifact((prev) => {
      if (!prev) return null;
      const existingVersions = prev.versions || [
        { version: 1, content: prev.content, timestamp: new Date() },
      ];
      const newVerNum = existingVersions.length + 1;
      const updatedVersions: ArtifactVersion[] = [
        ...existingVersions,
        { version: newVerNum, content, timestamp: new Date() },
      ];
      return {
        ...prev,
        content,
        versions: updatedVersions,
        currentVersionIndex: updatedVersions.length - 1,
      };
    });
  }, []);

  const setVersion = useCallback((versionIndex: number) => {
    setActiveArtifact((prev) => {
      if (!prev || !prev.versions || !prev.versions[versionIndex]) return prev;
      return {
        ...prev,
        content: prev.versions[versionIndex].content,
        currentVersionIndex: versionIndex,
      };
    });
  }, []);

  const restoreVersion = useCallback((versionIndex: number) => {
    setActiveArtifact((prev) => {
      if (!prev || !prev.versions || !prev.versions[versionIndex]) return prev;
      const targetContent = prev.versions[versionIndex].content;
      const newVerNum = prev.versions.length + 1;
      const updatedVersions: ArtifactVersion[] = [
        ...prev.versions,
        {
          version: newVerNum,
          content: targetContent,
          timestamp: new Date(),
          summary: `Restored from v${prev.versions[versionIndex].version}`,
        },
      ];
      return {
        ...prev,
        content: targetContent,
        versions: updatedVersions,
        currentVersionIndex: updatedVersions.length - 1,
      };
    });
  }, []);

  return (
    <ArtifactContext.Provider
      value={{
        activeArtifact,
        openArtifact,
        closeArtifact,
        updateArtifactContent,
        setVersion,
        restoreVersion,
      }}
    >
      {children}
    </ArtifactContext.Provider>
  );
};

export const useArtifact = (): ArtifactContextType => {
  const context = useContext(ArtifactContext);
  if (!context) {
    throw new Error('useArtifact must be used within an ArtifactProvider');
  }
  return context;
};

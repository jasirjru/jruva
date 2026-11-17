'use client';

import '@/styles/pages/plugins-projects.css';
import { useCallback, useEffect, useState } from 'react';
import { Switch } from '@astryxdesign/core/Switch';
import { PluginItem } from '@/types/workspace';
import { useAuth } from '@/context/AuthContext';
import { pluginCopy, pluginGroup } from './pluginCopy';
import { errorMessage, jsonBody, requestJson } from './workspaceApi';
import { SkeletonRows, WorkspaceNotice, WorkspacePageHead } from './WorkspacePageHead';

interface PluginsWorkspaceProps {
  onBackToChat: () => void;
}

type LoadState = 'loading' | 'ready' | 'error';

export function PluginsWorkspace({ onBackToChat }: PluginsWorkspaceProps) {
  const { isAuthenticated } = useAuth();
  const [plugins, setPlugins] = useState<PluginItem[]>([]);
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [loadError, setLoadError] = useState('');
  const [savingId, setSavingId] = useState<string | null>(null);
  const [rowError, setRowError] = useState<{ id: string; message: string } | null>(null);
  const [announcement, setAnnouncement] = useState('');

  const loadPlugins = useCallback(
    () =>
      requestJson<{ plugins?: PluginItem[] }>('/api/v1/plugins').then(
        (data) => {
          setPlugins(data.plugins ?? []);
          setLoadState('ready');
        },
        (err: unknown) => {
          setLoadError(errorMessage(err));
          setLoadState('error');
        },
      ),
    [],
  );

  useEffect(() => {
    void loadPlugins();
  }, [loadPlugins]);

  const retry = () => {
    setLoadState('loading');
    void loadPlugins();
  };

  const handleToggle = async (plugin: PluginItem, nextState: boolean) => {
    const name = pluginCopy(plugin).name;
    setSavingId(plugin.id);
    setRowError(null);
    try {
      await requestJson(`/api/v1/plugins/${plugin.id}/toggle`, jsonBody('POST', { is_enabled: nextState }));
      setPlugins((prev) => prev.map((p) => (p.id === plugin.id ? { ...p, is_enabled: nextState } : p)));
      setAnnouncement(`${name} is ${nextState ? 'on' : 'off'}.`);
    } catch (err) {
      setRowError({ id: plugin.id, message: `Could not save. ${errorMessage(err)}` });
    } finally {
      setSavingId(null);
    }
  };

  const working = plugins.filter((p) => pluginGroup(p) === 'working');
  const off = plugins.filter((p) => pluginGroup(p) === 'off');

  const renderRow = (plugin: PluginItem, isOff: boolean) => {
    const copy = pluginCopy(plugin);
    const { Icon } = copy;
    const error = rowError?.id === plugin.id ? rowError.message : null;
    return (
      <li key={plugin.id} className={`pp-plugin ${isOff ? 'is-off' : ''}`}>
        <span className="pp-plugin-icon" aria-hidden="true">
          <Icon size={18} strokeWidth={1.75} />
        </span>
        <div className="pp-plugin-body">
          <h3 className="pp-plugin-name">{copy.name}</h3>
          <p className="pp-plugin-desc">{copy.description}</p>
          {copy.hasSetting && !isOff && copy.settingNote && (
            <p className="pp-plugin-note">
              {isAuthenticated ? copy.settingNote : 'Sign in to turn this off for your account.'}
            </p>
          )}
          {error && (
            <p className="pp-field-error" role="alert">
              {error}
            </p>
          )}
        </div>
        <div className="pp-plugin-control">
          {copy.hasSetting && !isOff ? (
            <Switch
              label={copy.name}
              isLabelHidden
              value={plugin.is_enabled}
              isLoading={savingId === plugin.id}
              isDisabled={!isAuthenticated || savingId === plugin.id}
              onChange={(checked) => void handleToggle(plugin, checked)}
            />
          ) : isOff ? null : (
            <span className="pp-plugin-state">Always on</span>
          )}
        </div>
      </li>
    );
  };

  return (
    <div className="workspace-panel-container pp-page">
      <WorkspacePageHead
        title="Plugins"
        description="Tools Jruva can use while it answers."
        onBack={onBackToChat}
      />

      <span className="pp-visually-hidden" role="status">
        {announcement}
      </span>

      {loadState === 'loading' && <SkeletonRows count={4} label="Loading plugins" />}

      {loadState === 'error' && (
        <WorkspaceNotice tone="error" message={`Could not load plugins. ${loadError}`} actionLabel="Try again" onAction={retry} />
      )}

      {loadState === 'ready' && working.length === 0 && off.length === 0 && (
        <p className="pp-muted">No plugins are available on this server.</p>
      )}

      {loadState === 'ready' && working.length > 0 && (
        <section className="pp-section" aria-labelledby="plugins-working">
          <h2 id="plugins-working" className="pp-section-title">
            Available
          </h2>
          <ul className="pp-list">{working.map((p) => renderRow(p, false))}</ul>
        </section>
      )}

      {loadState === 'ready' && off.length > 0 && (
        <section className="pp-section" aria-labelledby="plugins-off">
          <h2 id="plugins-off" className="pp-section-title">
            Unavailable
          </h2>
          <ul className="pp-list">{off.map((p) => renderRow(p, true))}</ul>
        </section>
      )}
    </div>
  );
}

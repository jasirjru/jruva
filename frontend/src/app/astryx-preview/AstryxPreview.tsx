'use client';

import { useState } from 'react';
import { Theme } from '@astryxdesign/core/theme';
import { Button } from '@astryxdesign/core/Button';
import { TextInput } from '@astryxdesign/core/TextInput';
import { Badge } from '@astryxdesign/core/Badge';
import { Card } from '@astryxdesign/core/Card';
import { Switch } from '@astryxdesign/core/Switch';
import { Banner } from '@astryxdesign/core/Banner';
import { chocolateTheme } from '@astryxdesign/theme-chocolate/built';
import '@astryxdesign/theme-chocolate/fonts.css';

type Mode = 'light' | 'dark';
type Skin = 'jruva' | 'astryx';

const SAMPLE_ANSWER =
  'KTU S8 results are usually published 6 to 8 weeks after the last exam. Check ktu.edu.in under Results, or ask me to watch the page for you.';

function SampleScreen() {
  const [question, setQuestion] = useState('');
  const [isWebSearchOn, setIsWebSearchOn] = useState(true);

  return (
    <div style={{ display: 'grid', gap: 16, maxWidth: 640 }}>
      <Banner status="info" title="Jruva is warming up" description="The first answer can take up to 20 seconds." />

      <Card>
        <div style={{ display: 'grid', gap: 12, padding: 16 }}>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <Badge label="Web search" variant="info" />
            <Badge label="3 sources" variant="neutral" />
          </div>
          <p style={{ margin: 0, lineHeight: 1.6 }}>{SAMPLE_ANSWER}</p>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <Button label="Copy" variant="secondary" />
            <Button label="Regenerate" variant="ghost" />
            <Button label="Export PDF" variant="ghost" />
          </div>
        </div>
      </Card>

      <TextInput
        label="Ask Jruva"
        placeholder="Ask anything, in English or Malayalam"
        value={question}
        onChange={(value) => setQuestion(value)}
      />
      <Switch label="Search the web" value={isWebSearchOn} onChange={(checked) => setIsWebSearchOn(checked)} />
      <div style={{ display: 'flex', gap: 8 }}>
        <Button label="Send" variant="primary" />
        <Button label="Delete chat" variant="destructive" />
      </div>
    </div>
  );
}

export function AstryxPreview() {
  const [mode, setMode] = useState<Mode>('light');
  const [skin, setSkin] = useState<Skin>('jruva');
  const nextMode: Mode = mode === 'light' ? 'dark' : 'light';
  const nextSkin: Skin = skin === 'jruva' ? 'astryx' : 'jruva';
  const skinLabel = skin === 'jruva' ? 'Jruva colours on Astryx' : 'Pure Astryx chocolate';

  return (
    <Theme theme={chocolateTheme} mode={mode}>
      <div
        data-jruva-skin={skin === 'jruva' ? '' : undefined}
        style={{
          minHeight: '100dvh',
          padding: '24px 16px',
          background: 'var(--color-background-body)',
          color: 'var(--color-text-primary)',
          fontFamily: 'var(--font-family-body)',
        }}
      >
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 24 }}>
          <strong style={{ marginRight: 'auto' }} data-testid="skin-name">{skinLabel} · {mode}</strong>
          <Button label={`Show ${nextSkin === 'jruva' ? 'Jruva colours' : 'pure Astryx'}`} variant="secondary" onClick={() => setSkin(nextSkin)} />
          <Button label={`Switch to ${nextMode}`} variant="ghost" onClick={() => setMode(nextMode)} />
        </div>
        <SampleScreen />
      </div>
    </Theme>
  );
}

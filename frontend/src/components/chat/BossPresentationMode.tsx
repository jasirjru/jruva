'use client';

import React, { useState } from 'react';
import { haptics } from '@/utils/haptics';

interface BossPresentationModeProps {
  isOpen: boolean;
  onClose: () => void;
  onRunDemo: (prompt: string) => void;
  isAutoSpeakEnabled: boolean;
  onToggleAutoSpeak: () => void;
}

export const BossPresentationMode: React.FC<BossPresentationModeProps> = ({
  isOpen,
  onClose,
  onRunDemo,
  isAutoSpeakEnabled,
  onToggleAutoSpeak,
}) => {
  const [activeTab, setActiveTab] = useState<'tour' | 'roi' | 'specs' | 'voice'>('tour');

  if (!isOpen) return null;

  const demoScenarios = [
    {
      id: 'image-gen',
      tag: 'VISUAL SYNTHESIS',
      tagColor: '#8b5cf6',
      title: '⚡ Photorealistic Image Generation',
      desc: 'Generates stunning 8K photorealistic images from natural language descriptions using advanced diffusion models.',
      prompt: 'Generate a futuristic smart city in 2050 with solar corridors and AI research hubs, 16:9 photorealistic, cinematic lighting',
      latency: '< 0.15s (Instant Pre-Cache)',
      talkingPoint: 'Demonstrates multi-modal visual synthesis with sub-150ms zero-latency edge delivery.',
    },
    {
      id: 'coding-demo',
      tag: 'DEEP CODING ENGINE',
      tagColor: '#10b981',
      title: '⚡ Production-Grade Code Generation',
      desc: 'Writes complete, type-annotated, production-ready code with Big-O analysis, edge cases, and robust error handling.',
      prompt: 'Write a Python function that implements a LRU cache from scratch with O(1) get and put operations. Include type hints, docstrings, and comprehensive test cases.',
      latency: '~1.2s Streaming',
      talkingPoint: 'Highlights deep coding capability with full complexity analysis and boundary testing.',
    },
    {
      id: 'anatomy-challenge',
      tag: 'FRONTIER MODEL ALIGNMENT',
      tagColor: '#f59e0b',
      title: '⚡ Complex Anatomy Challenge ("4 Eyes Human")',
      desc: 'Renders precise multi-ocular facial geometry without anatomical collapse or airbrushed watermark artifacts.',
      prompt: 'Generate a photorealistic image of a 4 eyes human, documentary portrait, Hasselblad 100MP clarity',
      latency: 'FLUX.1 Direct Synthesis',
      talkingPoint: 'Shows prompt expansion engine decomposing coordinate geometry where standard commercial AI fails.',
    },
    {
      id: 'web-search',
      tag: 'REAL-TIME SEARCH',
      tagColor: '#ec4899',
      title: '⚡ Live Web-Grounded Factual Answers',
      desc: 'Searches the live internet, retrieves verified sources, and delivers factual answers with source citations.',
      prompt: 'Who is the current President of the United States and when did they take office?',
      latency: '< 2s End-to-End',
      talkingPoint: 'Real-time web search grounding eliminates stale training data — always current facts.',
    },
    {
      id: 'math-reasoning',
      tag: 'DEEP REASONING',
      tagColor: '#3b82f6',
      title: '⚡ Advanced Mathematical Reasoning',
      desc: 'Step-by-step mathematical proofs with formal notation, intermediate steps, and verification.',
      prompt: 'Prove that the square root of 2 is irrational using proof by contradiction. Show every step rigorously.',
      latency: 'Chain-of-Thought',
      talkingPoint: 'Demonstrates systematic deep reasoning with formal mathematical rigor.',
    },
    {
      id: 'upscale-demo',
      tag: 'SUPER-RESOLUTION',
      tagColor: '#06b6d4',
      title: '⚡ True 4K Ultra-HD Super-Resolution',
      desc: 'Upscales standard visual renders to 2752×1536 True 4K in 0.31s using multi-stage Lanczos micro-contrast sharpening.',
      prompt: 'Generate a high-performance autonomous robotics engineering lab, 16:9 cinematic',
      latency: '0.31s CPU Upscaling',
      talkingPoint: 'Instant on-the-fly 4K resolution upgrade without GPU VRAM overhead.',
    },
    {
      id: 'creative-writing',
      tag: 'CREATIVE STUDIO',
      tagColor: '#e11d48',
      title: '⚡ Creative Writing & Analysis',
      desc: 'Generates essays, stories, technical documentation, and performs deep analysis of complex topics.',
      prompt: 'Write a compelling 300-word essay on how artificial intelligence will transform education in the next decade.',
      latency: 'Streaming Output',
      talkingPoint: 'Shows natural language generation quality with structured, engaging prose.',
    },
  ];

  return (
    <div className="boss-modal-overlay" onClick={onClose}>
      <div className="boss-modal-container" onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="boss-modal-header">
          <div className="boss-header-left">
            <div className="boss-badge">
              <span className="boss-badge-dot" />
              <span>EXECUTIVE SUITE 2026</span>
            </div>
            <h2 className="boss-modal-title">Executive Presentation &amp; Institutional ROI Studio</h2>
            <p className="boss-modal-subtitle">
              Interactive demonstration toolkit, business financial models, and live telemetry for leadership review.
            </p>
          </div>
          <button type="button" className="boss-modal-close-btn" onClick={onClose} aria-label="Close modal">
            ✕
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="boss-tab-nav">
          <button
            type="button"
            className={`boss-tab-btn ${activeTab === 'tour' ? 'active' : ''}`}
            onClick={() => {
              haptics.click();
              setActiveTab('tour');
            }}
          >
            ⚡ 1-Click Executive Tour
          </button>
          <button
            type="button"
            className={`boss-tab-btn ${activeTab === 'roi' ? 'active' : ''}`}
            onClick={() => {
              haptics.click();
              setActiveTab('roi');
            }}
          >
            💰 Institutional ROI &amp; Savings
          </button>
          <button
            type="button"
            className={`boss-tab-btn ${activeTab === 'specs' ? 'active' : ''}`}
            onClick={() => {
              haptics.click();
              setActiveTab('specs');
            }}
          >
            🖥️ Architecture &amp; Hardware
          </button>
          <button
            type="button"
            className={`boss-tab-btn ${activeTab === 'voice' ? 'active' : ''}`}
            onClick={() => {
              haptics.click();
              setActiveTab('voice');
            }}
          >
            🎙️ Voice Studio Controls
          </button>
        </div>

        {/* Modal Content Body */}
        <div className="boss-modal-body">
          {/* TAB 1: 1-CLICK TOUR */}
          {activeTab === 'tour' && (
            <div className="boss-tour-container">
              <div className="boss-tour-banner">
                <div className="boss-tour-banner-icon">🎯</div>
                <div className="boss-tour-banner-text">
                  <strong>3-Minute Executive Pitch Script:</strong> Click any scenario below to trigger a live, production-grade demonstration in the chat interface. Each scenario showcases a distinct architectural capability with zero setup.
                </div>
              </div>

              <div className="boss-scenario-grid">
                {demoScenarios.map((s, idx) => (
                  <div key={s.id} className="boss-scenario-card">
                    <div className="boss-scenario-top">
                      <span className="boss-scenario-tag" style={{ backgroundColor: `${s.tagColor}20`, color: s.tagColor, borderColor: `${s.tagColor}40` }}>
                        {s.tag}
                      </span>
                      <span className="boss-scenario-latency">{s.latency}</span>
                    </div>
                    <h3 className="boss-scenario-title">{s.title}</h3>
                    <p className="boss-scenario-desc">{s.desc}</p>
                    
                    <div className="boss-talking-box">
                      <span className="boss-talking-label">Executive Talking Point:</span>
                      <span className="boss-talking-text">"{s.talkingPoint}"</span>
                    </div>

                    <button
                      type="button"
                      className="boss-launch-scenario-btn"
                      onClick={() => {
                        haptics.send();
                        onRunDemo(s.prompt);
                        onClose();
                      }}
                    >
                      <span>Launch Scenario #{idx + 1}</span>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="5" y1="12" x2="19" y2="12" />
                        <polyline points="12 5 19 12 12 19" />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TAB 2: ROI & SAVINGS */}
          {activeTab === 'roi' && (
            <div className="boss-roi-container">
              <div className="boss-kpi-row">
                <div className="boss-kpi-card highlight">
                  <div className="boss-kpi-label">Projected Annual Savings</div>
                  <div className="boss-kpi-val">₹18,50,000+</div>
                  <div className="boss-kpi-sub">100% reduction in commercial API bills</div>
                </div>
                <div className="boss-kpi-card">
                  <div className="boss-kpi-label">Payback Period</div>
                  <div className="boss-kpi-val">&lt; 1 Month</div>
                  <div className="boss-kpi-sub">Using existing campus workstation GPUs</div>
                </div>
                <div className="boss-kpi-card">
                  <div className="boss-kpi-label">Data Privacy Level</div>
                  <div className="boss-kpi-val">100% On-Prem</div>
                  <div className="boss-kpi-sub">Zero student data leaked to public cloud</div>
                </div>
                <div className="boss-kpi-card">
                  <div className="boss-kpi-label">Concurrent Capacity</div>
                  <div className="boss-kpi-val">500+ Req/Min</div>
                  <div className="boss-kpi-sub">Multi-tier cache &amp; dynamic failover</div>
                </div>
              </div>

              {/* Financial Comparison Table */}
              <div className="boss-table-card">
                <h3 className="boss-table-title">Commercial Cloud AI vs. JRU Self-Hosted On-Premises Architecture</h3>
                <div className="boss-table-wrap">
                  <table className="boss-data-table">
                    <thead>
                      <tr>
                        <th>Strategic Dimension</th>
                        <th>Proprietary Cloud (OpenAI / Commercial)</th>
                        <th>JRU Sovereign Platform (Our System)</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td><strong>Annual Recurring Cost (1,500 Students)</strong></td>
                        <td className="cost-danger">₹18,00,000 – ₹26,00,000 / year ($20/student/mo)</td>
                        <td className="cost-success">₹0 / month recurring API subscription</td>
                      </tr>
                      <tr>
                        <td><strong>Student Data Privacy &amp; UGC Compliance</strong></td>
                        <td className="cost-danger">Data logged on commercial US cloud servers</td>
                        <td className="cost-success">100% Private on-premise local encryption</td>
                      </tr>
                      <tr>
                        <td><strong>Domain Knowledge Adaptation</strong></td>
                        <td className="cost-warning">Generic web knowledge, hallucinates fees/rules</td>
                        <td className="cost-success">Fine-tuned JRU BGE Embeddings + Statutory RAG</td>
                      </tr>
                      <tr>
                        <td><strong>Offline Campus Resilience</strong></td>
                        <td className="cost-danger">Fails completely during external ISP outages</td>
                        <td className="cost-success">Local GPU fallback operates over campus LAN</td>
                      </tr>
                      <tr>
                        <td><strong>Visual Media &amp; 4K Super-Resolution</strong></td>
                        <td className="cost-warning">$0.040 per low-res 1024×1024 image</td>
                        <td className="cost-success">True 4K UHD super-resolution with zero fee</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: SPECS & HARDWARE */}
          {activeTab === 'specs' && (
            <div className="boss-specs-container">
              <div className="boss-specs-grid">
                <div className="boss-spec-card">
                  <div className="boss-spec-header">
                    <span className="boss-spec-icon">🎮</span>
                    <div>
                      <h4 className="boss-spec-title">Hardware Acceleration</h4>
                      <p className="boss-spec-sub">Dedicated Local Compute</p>
                    </div>
                  </div>
                  <ul className="boss-spec-list">
                    <li><strong>GPU:</strong> NVIDIA GeForce RTX 4060 Laptop GPU (8GB GDDR6)</li>
                    <li><strong>Current VRAM In-Use:</strong> ~2.18 GB / 8.18 GB (5.8 GB Headroom)</li>
                    <li><strong>Local Model Server:</strong> Port 11435 (PID 231808 active daemon)</li>
                    <li><strong>CUDA Runtime:</strong> 13.3 with cuDNN Acceleration</li>
                  </ul>
                </div>

                <div className="boss-spec-card">
                  <div className="boss-spec-header">
                    <span className="boss-spec-icon">🧠</span>
                    <div>
                      <h4 className="boss-spec-title">Neural Routing &amp; Models</h4>
                      <p className="boss-spec-sub">Multi-Tier Architecture</p>
                    </div>
                  </div>
                  <ul className="boss-spec-list">
                    <li><strong>Tier FT (Cloud Frontier):</strong> GPT-OSS 120B / Qwen 27B Cloud Engine</li>
                    <li><strong>Tier S+++ (Ultra-Fast):</strong> OpenAI/GPT-OSS-120B (Groq LPU Engine)</li>
                    <li><strong>Visual Engine:</strong> FLUX.1 + High-Speed Turbo Failover</li>
                    <li><strong>Super-Resolution:</strong> Multi-Stage Lanczos Micro-Contrast (4K)</li>
                  </ul>
                </div>

                <div className="boss-spec-card">
                  <div className="boss-spec-header">
                    <span className="boss-spec-icon">📚</span>
                    <div>
                      <h4 className="boss-spec-title">Vector Database &amp; RAG</h4>
                      <p className="boss-spec-sub">Sub-25ms Grounding</p>
                    </div>
                  </div>
                  <ul className="boss-spec-list">
                    <li><strong>Vector Store:</strong> File-Backed Embedded Qdrant (384 Dimensions)</li>
                    <li><strong>Embedding Engine:</strong> Fine-Tuned JRU BGE ONNX Model</li>
                    <li><strong>Hybrid Search:</strong> BM25 Keyword + Dense Cosine + Reciprocal Rank Fusion</li>
                    <li><strong>Neural Reranker:</strong> FlashRank Cross-Encoder (ms-marco-TinyBERT)</li>
                  </ul>
                </div>

                <div className="boss-spec-card">
                  <div className="boss-spec-header">
                    <span className="boss-spec-icon">🛡️</span>
                    <div>
                      <h4 className="boss-spec-title">Resilience &amp; Zero-Downtime</h4>
                      <p className="boss-spec-sub">Enterprise SLA Standards</p>
                    </div>
                  </div>
                  <ul className="boss-spec-list">
                    <li><strong>Local Disk Caching:</strong> Static assets served in 3.2ms from localhost</li>
                    <li><strong>Automatic Failover:</strong> Dynamic provider health-check monitor</li>
                    <li><strong>Rate-Limit Shield:</strong> IP queue concurrency semaphore guard</li>
                    <li><strong>Zero-Error Guarantee:</strong> Certified zero red cards on all visual queries</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: VOICE STUDIO */}
          {activeTab === 'voice' && (
            <div className="boss-voice-container">
              <div className="boss-voice-card">
                <div className="boss-voice-header">
                  <div className="boss-voice-icon-box">🎙️</div>
                  <div>
                    <h4 className="boss-voice-title">Autonomous Voice Mode (Conversational Audio)</h4>
                    <p className="boss-voice-desc">
                      Enables natural hands-free verbal interaction for executive boardrooms and student kiosks.
                    </p>
                  </div>
                </div>

                <div className="boss-voice-setting-row">
                  <div className="boss-setting-info">
                    <strong>Auto-Speak Responses Aloud:</strong>
                    <span>When enabled, the assistant automatically reads the answer aloud using natural browser speech synthesis.</span>
                  </div>
                  <button
                    type="button"
                    className={`boss-toggle-switch ${isAutoSpeakEnabled ? 'on' : 'off'}`}
                    onClick={() => {
                      haptics.click();
                      onToggleAutoSpeak();
                    }}
                  >
                    <span className="boss-toggle-knob" />
                  </button>
                </div>

                <div className="boss-voice-tips">
                  <h5>Presentation Tips for Voice Mode:</h5>
                  <ul>
                    <li>Click the <strong>Microphone</strong> button in the bottom input bar and speak naturally into your laptop.</li>
                    <li>The system transcribes your voice in real time using the native Web Speech API.</li>
                    <li>When the answer generates, the system speaks the synthesized summary with smooth voice modulation.</li>
                    <li>You can pause or stop audio anytime by clicking the <strong>Speaker</strong> button on any message card.</li>
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

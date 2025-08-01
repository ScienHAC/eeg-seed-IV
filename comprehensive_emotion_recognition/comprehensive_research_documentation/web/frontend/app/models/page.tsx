'use client';

import React, { useEffect, useState } from 'react';
import StageProgress from '../../components/StageProgress';
import PerformanceTable from '../../components/PerformanceTable';

interface ModelStage {
  stage: number;
  name: string;
  status: string;
  description: string;
  performance?: {
    accuracy: number;
    f1Score: number;
    trainingTime: number;
  };
  classificationReport?: Record<string, {
    emotion: string;
    precision: number;
    recall: number;
    f1Score: number;
    support: number;
  }>;
}

export default function Models() {
  const [stage1Data, setStage1Data] = useState<ModelStage | null>(null);
  const [stage2Data, setStage2Data] = useState<ModelStage | null>(null);

  useEffect(() => {
    // Load stage data
    Promise.all([
      fetch('/data/models_stage1.json').then(res => res.json()),
      fetch('/data/models_stage2.json').then(res => res.json())
    ]).then(([s1, s2]) => {
      setStage1Data(s1);
      setStage2Data(s2);
    }).catch(err => console.error('Error loading model data:', err));
  }, []);

  const stageNames = [
    'SVM Baseline',
    'Random Forest + SFS',
    'Autoencoder Features',
    'CNN Spatial',
    'LSTM Temporal',
    'Advanced Ensemble'
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="text-center">
            <h1 className="text-4xl font-bold mb-4">
              🤖 Model Architecture & Performance
            </h1>
            <p className="text-xl text-purple-100 mb-6">
              Six-Stage Progressive Development Approach
            </p>
            <div className="bg-white/10 rounded-lg px-6 py-3 inline-block">
              <span className="text-sm">
                From Traditional ML to Advanced Deep Learning
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Stage Progress */}
        <div className="mb-12">
          <StageProgress
            currentStage={3}
            totalStages={6}
            stageNames={stageNames}
            completedStages={[1, 2]}
          />
        </div>

        {/* Completed Stages */}
        <div className="space-y-8 mb-12">
          {/* Stage 1 */}
          {stage1Data && (
            <div className="research-card p-8">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">
                    Stage 1: {stage1Data.name}
                  </h2>
                  <p className="text-gray-600 mt-2">{stage1Data.description}</p>
                </div>
                <div className="text-right">
                  <div className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-medium">
                    ✓ Completed
                  </div>
                  <div className="text-2xl font-bold text-green-600 mt-2">
                    {stage1Data.performance?.accuracy}%
                  </div>
                </div>
              </div>

              <div className="grid md:grid-cols-3 gap-6 mb-6">
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">
                    {stage1Data.performance?.accuracy}%
                  </div>
                  <div className="text-sm text-gray-600">Accuracy</div>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">
                    {stage1Data.performance?.f1Score}%
                  </div>
                  <div className="text-sm text-gray-600">F1-Score</div>
                </div>
                <div className="text-center p-4 bg-orange-50 rounded-lg">
                  <div className="text-2xl font-bold text-orange-600">
                    {stage1Data.performance?.trainingTime}s
                  </div>
                  <div className="text-sm text-gray-600">Training Time</div>
                </div>
              </div>

              {stage1Data.classificationReport && (
                <PerformanceTable
                  data={Object.values(stage1Data.classificationReport).map(item => ({
                    emotion: item.emotion,
                    precision: item.precision,
                    recall: item.recall,
                    f1Score: item.f1Score,
                    support: item.support
                  }))}
                  title="Per-Class Performance - Stage 1"
                />
              )}
            </div>
          )}

          {/* Stage 2 */}
          {stage2Data && (
            <div className="research-card p-8 border-2 border-green-200">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">
                    Stage 2: {stage2Data.name} ⭐
                  </h2>
                  <p className="text-gray-600 mt-2">{stage2Data.description}</p>
                </div>
                <div className="text-right">
                  <div className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-medium">
                    ✓ Breakthrough
                  </div>
                  <div className="text-3xl font-bold text-green-600 mt-2">
                    {stage2Data.performance?.accuracy}%
                  </div>
                </div>
              </div>

              <div className="grid md:grid-cols-3 gap-6 mb-6">
                <div className="text-center p-4 bg-green-50 rounded-lg border-2 border-green-200">
                  <div className="text-3xl font-bold text-green-600">
                    {stage2Data.performance?.accuracy}%
                  </div>
                  <div className="text-sm text-gray-600">Accuracy</div>
                  <div className="text-xs text-green-600 mt-1">Breakthrough!</div>
                </div>
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">
                    {stage2Data.performance?.f1Score}%
                  </div>
                  <div className="text-sm text-gray-600">F1-Score</div>
                </div>
                <div className="text-center p-4 bg-orange-50 rounded-lg">
                  <div className="text-2xl font-bold text-orange-600">
                    {Math.round((stage2Data.performance?.trainingTime || 0) / 60)}m
                  </div>
                  <div className="text-sm text-gray-600">Training Time</div>
                </div>
              </div>

              {stage2Data.classificationReport && (
                <PerformanceTable
                  data={Object.values(stage2Data.classificationReport).map(item => ({
                    emotion: item.emotion,
                    precision: item.precision,
                    recall: item.recall,
                    f1Score: item.f1Score,
                    support: item.support
                  }))}
                  title="Per-Class Performance - Stage 2 (97.7% Accuracy)"
                />
              )}

              <div className="mt-6 p-4 bg-green-50 rounded-lg">
                <h3 className="font-semibold text-green-800 mb-2">Key Breakthrough Features:</h3>
                <ul className="text-sm text-green-700 space-y-1">
                  <li>• Sequential Feature Selection: 310 → 60 features</li>
                  <li>• Random Forest with 500 estimators</li>
                  <li>• Advanced hyperparameter optimization</li>
                  <li>• 20.06% improvement over baseline</li>
                </ul>
              </div>
            </div>
          )}
        </div>

        {/* Planned Stages */}
        <div className="research-card p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            🚀 Planned Development Stages (3-6)
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="border border-gray-200 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-900">Stage 3: Autoencoder Features</h3>
                <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">Planned</span>
              </div>
              <p className="text-gray-600 text-sm mb-4">
                Deep autoencoder for unsupervised feature learning and dimensionality reduction.
              </p>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500">Target Accuracy:</span>
                  <span className="font-medium">95-98%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Features:</span>
                  <span className="font-medium">310 → 32</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Architecture:</span>
                  <span className="font-medium">Encoder-Decoder</span>
                </div>
              </div>
            </div>

            <div className="border border-gray-200 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-900">Stage 4: CNN Spatial</h3>
                <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">Planned</span>
              </div>
              <p className="text-gray-600 text-sm mb-4">
                2D CNN treating EEG as spatial-spectral images with attention mechanisms.
              </p>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500">Target Accuracy:</span>
                  <span className="font-medium">96-99%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Input Shape:</span>
                  <span className="font-medium">(62, 5, 1)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Layers:</span>
                  <span className="font-medium">Conv2D + Attention</span>
                </div>
              </div>
            </div>

            <div className="border border-gray-200 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-900">Stage 5: LSTM Temporal</h3>
                <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">Planned</span>
              </div>
              <p className="text-gray-600 text-sm mb-4">
                LSTM for capturing temporal dependencies and emotion evolution patterns.
              </p>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500">Target Accuracy:</span>
                  <span className="font-medium">96-99%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Sequence Length:</span>
                  <span className="font-medium">Variable</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Architecture:</span>
                  <span className="font-medium">Bidirectional LSTM</span>
                </div>
              </div>
            </div>

            <div className="border border-gray-200 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-900">Stage 6: Advanced Ensemble</h3>
                <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">Planned</span>
              </div>
              <p className="text-gray-600 text-sm mb-4">
                Multi-model ensemble combining best performers from all previous stages.
              </p>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500">Target Accuracy:</span>
                  <span className="font-medium">&gt;98%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Models:</span>
                  <span className="font-medium">RF + AE + CNN + LSTM</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Voting:</span>
                  <span className="font-medium">Weighted</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Model Comparison */}
        <div className="research-card p-8 mt-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            📊 Model Comparison & Analysis
          </h2>
          
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Stage</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Method</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Accuracy</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Features</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Time</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                <tr className="hover:bg-gray-50">
                  <td className="px-6 py-4 text-sm font-medium text-gray-900">1</td>
                  <td className="px-6 py-4 text-sm text-gray-900">SVM</td>
                  <td className="px-6 py-4 text-sm text-gray-900">77.64%</td>
                  <td className="px-6 py-4 text-sm text-gray-900">310</td>
                  <td className="px-6 py-4 text-sm text-gray-900">31s</td>
                  <td className="px-6 py-4"><span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs">✓ Done</span></td>
                </tr>
                <tr className="hover:bg-gray-50 bg-green-50">
                  <td className="px-6 py-4 text-sm font-medium text-gray-900">2</td>
                  <td className="px-6 py-4 text-sm text-gray-900">Random Forest + SFS</td>
                  <td className="px-6 py-4 text-sm font-bold text-green-600">97.70%</td>
                  <td className="px-6 py-4 text-sm text-gray-900">60</td>
                  <td className="px-6 py-4 text-sm text-gray-900">28m</td>
                  <td className="px-6 py-4"><span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs">✓ Best</span></td>
                </tr>
                <tr className="hover:bg-gray-50">
                  <td className="px-6 py-4 text-sm font-medium text-gray-900">3</td>
                  <td className="px-6 py-4 text-sm text-gray-900">Autoencoder</td>
                  <td className="px-6 py-4 text-sm text-gray-500">95-98%</td>
                  <td className="px-6 py-4 text-sm text-gray-900">32</td>
                  <td className="px-6 py-4 text-sm text-gray-500">TBD</td>
                  <td className="px-6 py-4"><span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">Planned</span></td>
                </tr>
                <tr className="hover:bg-gray-50">
                  <td className="px-6 py-4 text-sm font-medium text-gray-900">4</td>
                  <td className="px-6 py-4 text-sm text-gray-900">CNN</td>
                  <td className="px-6 py-4 text-sm text-gray-500">96-99%</td>
                  <td className="px-6 py-4 text-sm text-gray-900">Variable</td>
                  <td className="px-6 py-4 text-sm text-gray-500">TBD</td>
                  <td className="px-6 py-4"><span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">Planned</span></td>
                </tr>
                <tr className="hover:bg-gray-50">
                  <td className="px-6 py-4 text-sm font-medium text-gray-900">5</td>
                  <td className="px-6 py-4 text-sm text-gray-900">LSTM</td>
                  <td className="px-6 py-4 text-sm text-gray-500">96-99%</td>
                  <td className="px-6 py-4 text-sm text-gray-900">Variable</td>
                  <td className="px-6 py-4 text-sm text-gray-500">TBD</td>
                  <td className="px-6 py-4"><span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">Planned</span></td>
                </tr>
                <tr className="hover:bg-gray-50">
                  <td className="px-6 py-4 text-sm font-medium text-gray-900">6</td>
                  <td className="px-6 py-4 text-sm text-gray-900">Ensemble</td>
                  <td className="px-6 py-4 text-sm text-gray-500">&gt;98%</td>
                  <td className="px-6 py-4 text-sm text-gray-900">Combined</td>
                  <td className="px-6 py-4 text-sm text-gray-500">TBD</td>
                  <td className="px-6 py-4"><span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">Planned</span></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

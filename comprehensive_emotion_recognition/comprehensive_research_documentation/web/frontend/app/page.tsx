'use client';

import React, { useEffect, useState } from 'react';
import StageProgress from '../components/StageProgress';

interface DatasetInfo {
  name: string;
  totalSubjects: number;
  sessions: { total: number };
  trials: { totalTrials: number };
  emotions: Array<{ name: string; color: string }>;
}

export default function Home() {
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);

  useEffect(() => {
    fetch('/data/dataset.json')
      .then(res => res.json())
      .then(data => setDatasetInfo(data))
      .catch(err => console.error('Error loading dataset info:', err));
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
      <div className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center">
            <h1 className="text-4xl font-bold mb-4">
              🧠 EEG-Based Emotion Recognition
            </h1>
            <p className="text-xl text-blue-100 mb-6">
              Achieving 97.7% Accuracy with SEED-IV Dataset
            </p>
            <div className="flex justify-center space-x-8 text-sm">
              <div className="bg-white/10 rounded-lg px-4 py-2">
                <div className="text-2xl font-bold">97.7%</div>
                <div>Best Accuracy</div>
              </div>
              <div className="bg-white/10 rounded-lg px-4 py-2">
                <div className="text-2xl font-bold">15</div>
                <div>Subjects</div>
              </div>
              <div className="bg-white/10 rounded-lg px-4 py-2">
                <div className="text-2xl font-bold">1,080</div>
                <div>Samples</div>
              </div>
              <div className="bg-white/10 rounded-lg px-4 py-2">
                <div className="text-2xl font-bold">4</div>
                <div>Emotions</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Executive Summary */}
        <div className="mb-12">
          <div className="research-card p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              📊 Executive Summary
            </h2>
            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <p className="text-gray-700 mb-4">
                  This research presents a comprehensive six-stage approach to EEG-based emotion 
                  recognition using the SEED-IV dataset. Our methodology progresses from traditional 
                  machine learning to advanced deep learning techniques.
                </p>
                <div className="space-y-2">
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm">Stage 1: SVM Baseline (77.64% accuracy)</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm">Stage 2: Random Forest + SFS (97.70% accuracy)</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                    <span className="text-sm text-gray-500">Stages 3-6: Advanced deep learning (planned)</span>
                  </div>
                </div>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 mb-3">Key Achievements</h3>
                <ul className="space-y-2 text-sm text-gray-700">
                  <li>• 97.7% accuracy breakthrough with Random Forest</li>
                  <li>• 20.06% improvement through feature selection</li>
                  <li>• Optimal 60-feature subset from 310 features</li>
                  <li>• Robust cross-validation performance</li>
                  <li>• Complete reproducible research framework</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Stage Progress */}
        <div className="mb-12">
          <StageProgress
            currentStage={3}
            totalStages={6}
            stageNames={stageNames}
            completedStages={[1, 2]}
          />
        </div>

        {/* Quick Navigation */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          <a href="/dataset" className="research-card p-6 hover:shadow-xl transition-all duration-300">
            <div className="text-3xl mb-3">📊</div>
            <h3 className="font-semibold text-gray-900 mb-2">Dataset</h3>
            <p className="text-sm text-gray-600">SEED-IV specifications, 15 subjects, 4 emotions</p>
          </a>
          
          <a href="/features" className="research-card p-6 hover:shadow-xl transition-all duration-300">
            <div className="text-3xl mb-3">🔬</div>
            <h3 className="font-semibold text-gray-900 mb-2">Features</h3>
            <p className="text-sm text-gray-600">Multi-domain feature engineering, 310→60 optimization</p>
          </a>
          
          <a href="/models" className="research-card p-6 hover:shadow-xl transition-all duration-300">
            <div className="text-3xl mb-3">🤖</div>
            <h3 className="font-semibold text-gray-900 mb-2">Models</h3>
            <p className="text-sm text-gray-600">Six-stage architecture, SVM to ensemble methods</p>
          </a>
          
          <a href="/results" className="research-card p-6 hover:shadow-xl transition-all duration-300">
            <div className="text-3xl mb-3">📈</div>
            <h3 className="font-semibold text-gray-900 mb-2">Results</h3>
            <p className="text-sm text-gray-600">97.7% accuracy, performance analysis, benchmarks</p>
          </a>
        </div>

        {/* Dataset Overview */}
        {datasetInfo && (
          <div className="research-card p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">
              🗂️ Dataset Overview: {datasetInfo.name}
            </h2>
            <div className="grid md:grid-cols-4 gap-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600 mb-2">
                  {datasetInfo.totalSubjects}
                </div>
                <div className="text-sm text-gray-500">Subjects</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-green-600 mb-2">
                  {datasetInfo.sessions.total}
                </div>
                <div className="text-sm text-gray-500">Sessions</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-600 mb-2">
                  {datasetInfo.trials.totalTrials}
                </div>
                <div className="text-sm text-gray-500">Total Trials</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-red-600 mb-2">
                  {datasetInfo.emotions.length}
                </div>
                <div className="text-sm text-gray-500">Emotions</div>
              </div>
            </div>
            
            <div className="mt-6 flex justify-center space-x-4">
              {datasetInfo.emotions.map((emotion, index) => (
                <span
                  key={index}
                  className="emotion-badge"
                  style={{
                    backgroundColor: `${emotion.color}20`,
                    color: emotion.color
                  }}
                >
                  {emotion.name}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

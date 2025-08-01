'use client';

import React, { useEffect, useState } from 'react';

interface DatasetInfo {
  name: string;
  description: string;
  origin: string;
  publication: string;
  totalSubjects: number;
  genderDistribution: { male: number; female: number };
  ageRange: { min: number; max: number; mean: number; std: number };
  sessions: { total: number; spacing: string };
  trials: { perSession: number; totalTrials: number };
  emotions: Array<{
    label: number;
    name: string;
    description: string;
    arousal: string;
    valence: string;
    clipsPerSession: number;
    color: string;
  }>;
  eegSpecs: {
    channels: number;
    system: string;
    samplingRate: number;
    preprocessing: {
      bandpassFilter: string;
      notchFilter: string;
      artifactRemoval: string;
      reference: string;
    };
    trialDuration: string;
  };
  frequencyBands: Array<{
    name: string;
    symbol: string;
    range: string;
    significance: string;
    color: string;
  }>;
  dataStructure: {
    originalFormat: string;
    convertedFormat: string;
    features: {
      total: number;
      calculation: string;
      types: string[];
    };
    sessionLabels: Record<string, number[]>;
  };
  sampleDistribution: {
    total: number;
    perEmotion: number;
    breakdown: Record<string, string>;
  };
}

export default function Dataset() {
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);

  useEffect(() => {
    fetch('/data/dataset.json')
      .then(res => res.json())
      .then(data => setDatasetInfo(data))
      .catch(err => console.error('Error loading dataset info:', err));
  }, []);

  if (!datasetInfo) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading dataset information...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-600 to-blue-600 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="text-center">
            <h1 className="text-4xl font-bold mb-4">
              📊 {datasetInfo.name}
            </h1>
            <p className="text-xl text-green-100 mb-6">
              {datasetInfo.description}
            </p>
            <div className="bg-white/10 rounded-lg px-6 py-3 inline-block">
              <span className="text-sm">
                📚 {datasetInfo.publication}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Overview Stats */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          <div className="research-card p-6 text-center">
            <div className="text-3xl font-bold text-blue-600 mb-2">
              {datasetInfo.totalSubjects}
            </div>
            <div className="text-sm text-gray-500">Total Subjects</div>
            <div className="text-xs text-gray-400 mt-1">
              {datasetInfo.genderDistribution.male}M / {datasetInfo.genderDistribution.female}F
            </div>
          </div>
          <div className="research-card p-6 text-center">
            <div className="text-3xl font-bold text-green-600 mb-2">
              {datasetInfo.sessions.total}
            </div>
            <div className="text-sm text-gray-500">Sessions</div>
            <div className="text-xs text-gray-400 mt-1">
              {datasetInfo.sessions.spacing}
            </div>
          </div>
          <div className="research-card p-6 text-center">
            <div className="text-3xl font-bold text-purple-600 mb-2">
              {datasetInfo.trials.totalTrials}
            </div>
            <div className="text-sm text-gray-500">Total Trials</div>
            <div className="text-xs text-gray-400 mt-1">
              {datasetInfo.trials.perSession} per session
            </div>
          </div>
          <div className="research-card p-6 text-center">
            <div className="text-3xl font-bold text-red-600 mb-2">
              {datasetInfo.emotions.length}
            </div>
            <div className="text-sm text-gray-500">Emotions</div>
            <div className="text-xs text-gray-400 mt-1">
              Balanced distribution
            </div>
          </div>
        </div>

        {/* Demographics */}
        <div className="research-card p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            👥 Demographics & Study Design
          </h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="font-semibold text-gray-900 mb-3">Participant Information</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Age Range:</span>
                  <span className="font-medium">
                    {datasetInfo.ageRange.min}-{datasetInfo.ageRange.max} years
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Mean Age:</span>
                  <span className="font-medium">
                    {datasetInfo.ageRange.mean} ± {datasetInfo.ageRange.std} years
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Gender Distribution:</span>
                  <span className="font-medium">
                    {datasetInfo.genderDistribution.male} Male, {datasetInfo.genderDistribution.female} Female
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Origin:</span>
                  <span className="font-medium">{datasetInfo.origin}</span>
                </div>
              </div>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-3">Study Protocol</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Sessions:</span>
                  <span className="font-medium">{datasetInfo.sessions.total} sessions</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Session Spacing:</span>
                  <span className="font-medium">{datasetInfo.sessions.spacing}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Trials per Session:</span>
                  <span className="font-medium">{datasetInfo.trials.perSession}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Trial Duration:</span>
                  <span className="font-medium">{datasetInfo.eegSpecs.trialDuration}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Emotions */}
        <div className="research-card p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            😊 Emotional Categories
          </h2>
          <div className="grid md:grid-cols-2 gap-6">
            {datasetInfo.emotions.map((emotion) => (
              <div
                key={emotion.label}
                className="border rounded-lg p-4"
                style={{ borderColor: emotion.color }}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div
                      className="w-4 h-4 rounded-full"
                      style={{ backgroundColor: emotion.color }}
                    ></div>
                    <h3 className="font-semibold text-gray-900">
                      {emotion.name} (Label: {emotion.label})
                    </h3>
                  </div>
                  <span className="text-sm text-gray-500">
                    {emotion.clipsPerSession} clips/session
                  </span>
                </div>
                <p className="text-gray-700 text-sm mb-3">{emotion.description}</p>
                <div className="flex justify-between text-xs">
                  <span className="bg-gray-100 px-2 py-1 rounded">
                    Arousal: {emotion.arousal}
                  </span>
                  <span className="bg-gray-100 px-2 py-1 rounded">
                    Valence: {emotion.valence}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* EEG Specifications */}
        <div className="research-card p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            🧠 EEG Recording Specifications
          </h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="font-semibold text-gray-900 mb-3">Hardware & Setup</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Channels:</span>
                  <span className="font-medium">{datasetInfo.eegSpecs.channels}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">System:</span>
                  <span className="font-medium">{datasetInfo.eegSpecs.system}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Sampling Rate:</span>
                  <span className="font-medium">{datasetInfo.eegSpecs.samplingRate} Hz</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Reference:</span>
                  <span className="font-medium">{datasetInfo.eegSpecs.preprocessing.reference}</span>
                </div>
              </div>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-3">Preprocessing</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Bandpass Filter:</span>
                  <span className="font-medium">{datasetInfo.eegSpecs.preprocessing.bandpassFilter}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Notch Filter:</span>
                  <span className="font-medium">{datasetInfo.eegSpecs.preprocessing.notchFilter}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Artifact Removal:</span>
                  <span className="font-medium">{datasetInfo.eegSpecs.preprocessing.artifactRemoval}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Frequency Bands */}
        <div className="research-card p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            📊 Frequency Band Analysis
          </h2>
          <div className="grid gap-4">
            {datasetInfo.frequencyBands.map((band, index) => (
              <div
                key={index}
                className="flex items-center p-4 rounded-lg border"
                style={{ borderColor: band.color }}
              >
                <div
                  className="w-12 h-12 rounded-full flex items-center justify-center text-white font-bold mr-4"
                  style={{ backgroundColor: band.color }}
                >
                  {band.symbol}
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold text-gray-900">
                      {band.name} ({band.range})
                    </h3>
                  </div>
                  <p className="text-gray-600 text-sm mt-1">{band.significance}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Data Structure */}
        <div className="research-card p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            🗂️ Data Structure & Format
          </h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="font-semibold text-gray-900 mb-3">File Structure</h3>
              <div className="bg-gray-50 rounded-lg p-4 font-mono text-sm">
                <div className="text-gray-600 mb-2">Original: {datasetInfo.dataStructure.originalFormat}</div>
                <div className="text-gray-600 mb-2">Converted: {datasetInfo.dataStructure.convertedFormat}</div>
                <div className="text-gray-900 font-semibold">
                  Total Features: {datasetInfo.dataStructure.features.total}
                </div>
                <div className="text-gray-600 text-xs mt-1">
                  ({datasetInfo.dataStructure.features.calculation})
                </div>
              </div>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-3">Feature Types</h3>
              <div className="space-y-2">
                {datasetInfo.dataStructure.features.types.map((type, index) => (
                  <div key={index} className="bg-blue-50 rounded px-3 py-2 text-sm">
                    <span className="font-medium text-blue-900">{type}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          <div className="mt-6 pt-6 border-t border-gray-200">
            <h3 className="font-semibold text-gray-900 mb-3">Sample Distribution</h3>
            <div className="grid md:grid-cols-3 gap-6 text-center">
              <div>
                <div className="text-2xl font-bold text-blue-600">
                  {datasetInfo.sampleDistribution.total}
                </div>
                <div className="text-sm text-gray-500">Total Samples</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-green-600">
                  {datasetInfo.sampleDistribution.perEmotion}
                </div>
                <div className="text-sm text-gray-500">Per Emotion</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-purple-600">
                  {datasetInfo.dataStructure.features.total}
                </div>
                <div className="text-sm text-gray-500">Features</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

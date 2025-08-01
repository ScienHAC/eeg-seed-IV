'use client';

import React, { useState, useEffect } from 'react';

interface FeaturesData {
  total_features: number;
  selected_features: number;
  current_band?: string;
  band_specific?: boolean;
}

export default function Features() {
  const [selectedBand, setSelectedBand] = useState('all');
  const [selectedEmotion, setSelectedEmotion] = useState('all');
  const [selectedSubject, setSelectedSubject] = useState('all');
  const [featuresData, setFeaturesData] = useState<FeaturesData | null>(null);

  useEffect(() => {
    // Load features data
    fetch('/data/features.json')
      .then(response => response.json())
      .then(data => setFeaturesData(data))
      .catch(error => console.error('Error loading features data:', error));
  }, []);

  const frequencyBands = [
    { value: 'all', label: 'All Bands' },
    { value: 'delta', label: 'Delta (1-4 Hz)' },
    { value: 'theta', label: 'Theta (4-8 Hz)' },
    { value: 'alpha', label: 'Alpha (8-14 Hz)' },
    { value: 'beta', label: 'Beta (14-31 Hz)' },
    { value: 'gamma', label: 'Gamma (31-50 Hz)' }
  ];

  const emotions = [
    { value: 'all', label: 'All Emotions' },
    { value: 'happy', label: 'Happy' },
    { value: 'sad', label: 'Sad' },
    { value: 'fear', label: 'Fear' },
    { value: 'neutral', label: 'Neutral' }
  ];

  const subjects = Array.from({ length: 15 }, (_, i) => ({
    value: i === 0 ? 'all' : `subject_${i}`,
    label: i === 0 ? 'All Subjects' : `Subject ${i}`
  }));

  const getFilteredData = () => {
    if (!featuresData) return null;

    // Simulate data filtering based on selections
    let filtered: FeaturesData = { ...featuresData };
    
    if (selectedBand !== 'all') {
      // Filter by frequency band
      filtered = {
        ...filtered,
        current_band: selectedBand,
        band_specific: true
      };
    }

    return filtered;
  };

  const featureCategories = [
    {
      name: 'Spectral Features',
      count: 310,
      description: 'Power spectral density across frequency bands',
      features: [
        'Band Power (Delta, Theta, Alpha, Beta, Gamma)',
        'Relative Power per frequency band',
        'Power Spectral Entropy',
        'Spectral Edge Frequency',
        'Peak Frequency per band'
      ],
      importance: 0.34
    },
    {
      name: 'Statistical Features',
      count: 186,
      description: 'Time-domain statistical measures',
      features: [
        'Mean, Variance, Standard Deviation',
        'Skewness and Kurtosis',
        'Range and Interquartile Range',
        'Zero Crossing Rate',
        'Peak-to-Peak Amplitude'
      ],
      importance: 0.28
    },
    {
      name: 'Connectivity Features',
      count: 248,
      description: 'Inter-channel relationships and coherence',
      features: [
        'Coherence between electrode pairs',
        'Phase Lag Index',
        'Cross-correlation coefficients',
        'Mutual Information',
        'Transfer Entropy'
      ],
      importance: 0.23
    },
    {
      name: 'Complexity Features',
      count: 124,
      description: 'Non-linear dynamics and entropy measures',
      features: [
        'Sample Entropy',
        'Approximate Entropy',
        'Lempel-Ziv Complexity',
        'Higuchi Fractal Dimension',
        'Detrended Fluctuation Analysis'
      ],
      importance: 0.15
    }
  ];

  const topFeatures = [
    { name: 'Gamma Band Power (Fp1)', importance: 0.089, type: 'Spectral' },
    { name: 'Alpha-Beta Coherence (F3-F4)', importance: 0.076, type: 'Connectivity' },
    { name: 'Theta Band Entropy (T8)', importance: 0.071, type: 'Spectral' },
    { name: 'Sample Entropy (Cz)', importance: 0.068, type: 'Complexity' },
    { name: 'Beta Power Asymmetry (F3-F4)', importance: 0.065, type: 'Spectral' },
    { name: 'Cross-correlation (Fp1-Fp2)', importance: 0.062, type: 'Connectivity' },
    { name: 'Delta Band Variance (O1)', importance: 0.059, type: 'Statistical' },
    { name: 'Higuchi FD (P4)', importance: 0.056, type: 'Complexity' },
    { name: 'Alpha Band Skewness (C3)', importance: 0.053, type: 'Statistical' },
    { name: 'Gamma-Delta Ratio (Fz)', importance: 0.051, type: 'Spectral' }
  ];

  const channelData = {
    frontal: {
      channels: ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz'],
      importance: 0.42,
      emotions: ['Happy', 'Neutral'],
      description: 'Executive functions, attention, emotional regulation'
    },
    central: {
      channels: ['C3', 'C4', 'Cz'],
      importance: 0.28,
      emotions: ['Fear', 'Sad'],
      description: 'Motor cortex, sensory processing'
    },
    temporal: {
      channels: ['T7', 'T8'],
      importance: 0.18,
      emotions: ['Happy', 'Sad'],
      description: 'Auditory processing, memory, emotion'
    },
    parietal: {
      channels: ['P3', 'P4', 'Pz'],
      importance: 0.08,
      emotions: ['Neutral'],
      description: 'Spatial processing, attention'
    },
    occipital: {
      channels: ['O1', 'O2'],
      importance: 0.04,
      emotions: ['Fear'],
      description: 'Visual processing, visual attention'
    }
  };

  if (!featuresData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading feature analysis data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-600 to-teal-600 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="text-center">
            <h1 className="text-4xl font-bold mb-4">
              🧠 Feature Engineering & Analysis
            </h1>
            <p className="text-xl text-emerald-100 mb-6">
              Interactive exploration of EEG features driving 97.7% accuracy
            </p>
            <div className="bg-white/10 rounded-lg px-6 py-3 inline-block">
              <span className="text-sm">
                {featuresData?.total_features || 868} engineered features | 
                {featuresData?.selected_features || 15} optimal features selected
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Interactive Filters */}
        <div className="research-card p-6 mb-8">
          <h2 className="text-xl font-bold text-gray-900 mb-4">
            🔍 Interactive Data Explorer
          </h2>
          <div className="grid md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Frequency Band
              </label>
              <select
                value={selectedBand}
                onChange={(e) => setSelectedBand(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500"
              >
                {frequencyBands.map(band => (
                  <option key={band.value} value={band.value}>
                    {band.label}
                  </option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Emotion State
              </label>
              <select
                value={selectedEmotion}
                onChange={(e) => setSelectedEmotion(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500"
              >
                {emotions.map(emotion => (
                  <option key={emotion.value} value={emotion.value}>
                    {emotion.label}
                  </option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Subject
              </label>
              <select
                value={selectedSubject}
                onChange={(e) => setSelectedSubject(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500"
              >
                {subjects.map(subject => (
                  <option key={subject.value} value={subject.value}>
                    {subject.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
          
          <div className="mt-4 p-3 bg-emerald-50 rounded-lg">
            <p className="text-sm text-emerald-800">
              <strong>Active Filters:</strong> {selectedBand !== 'all' && `${selectedBand.toUpperCase()} band | `}
              {selectedEmotion !== 'all' && `${selectedEmotion} emotion | `}
              {selectedSubject !== 'all' && `${selectedSubject.replace('_', ' ')} | `}
              {selectedBand === 'all' && selectedEmotion === 'all' && selectedSubject === 'all' && 'All data selected'}
            </p>
          </div>
        </div>

        {/* Feature Categories Overview */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {featureCategories.map((category, index) => (
            <div key={index} className="research-card p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-900">{category.name}</h3>
                <div className="text-2xl font-bold text-emerald-600">{category.count}</div>
              </div>
              
              <p className="text-sm text-gray-600 mb-4">{category.description}</p>
              
              <div className="mb-4">
                <div className="flex justify-between text-xs text-gray-500 mb-1">
                  <span>Importance</span>
                  <span>{(category.importance * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-emerald-600 h-2 rounded-full"
                    style={{ width: `${category.importance * 100}%` }}
                  ></div>
                </div>
              </div>
              
              <ul className="text-xs text-gray-600 space-y-1">
                {category.features.slice(0, 3).map((feature, idx) => (
                  <li key={idx} className="flex items-start">
                    <span className="w-1 h-1 bg-gray-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                    {feature}
                  </li>
                ))}
                {category.features.length > 3 && (
                  <li className="text-gray-400">+{category.features.length - 3} more...</li>
                )}
              </ul>
            </div>
          ))}
        </div>

        {/* Top Selected Features */}
        <div className="research-card p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            🏆 Top Selected Features (Random Forest + SFS)
          </h2>
          
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-3 px-4 font-semibold text-gray-900">Rank</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-900">Feature Name</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-900">Type</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-900">Importance</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-900">Contribution</th>
                </tr>
              </thead>
              <tbody>
                {topFeatures.map((feature, index) => (
                  <tr key={index} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-3 px-4">
                      <div className="flex items-center">
                        <span className="bg-emerald-100 text-emerald-800 px-2 py-1 rounded text-sm font-medium">
                          #{index + 1}
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-4 font-medium text-gray-900">{feature.name}</td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-1 rounded text-xs ${
                        feature.type === 'Spectral' ? 'bg-blue-100 text-blue-800' :
                        feature.type === 'Statistical' ? 'bg-green-100 text-green-800' :
                        feature.type === 'Connectivity' ? 'bg-purple-100 text-purple-800' :
                        'bg-orange-100 text-orange-800'
                      }`}>
                        {feature.type}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center">
                        <div className="w-24 bg-gray-200 rounded-full h-2 mr-3">
                          <div
                            className="bg-emerald-600 h-2 rounded-full"
                            style={{ width: `${feature.importance * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-sm text-gray-600">{(feature.importance * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-600">
                      {(feature.importance * 976).toFixed(1)} accuracy points
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Channel Analysis */}
        <div className="research-card p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            📍 EEG Channel Analysis
          </h2>
          
          <div className="grid lg:grid-cols-2 gap-8">
            <div>
              <h3 className="font-semibold text-gray-900 mb-4">Channel Importance by Region</h3>
              <div className="space-y-4">
                {Object.entries(channelData).map(([region, data]) => (
                  <div key={region} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <h4 className="font-medium text-gray-900 capitalize">{region}</h4>
                        <p className="text-sm text-gray-600">{data.description}</p>
                      </div>
                      <span className="bg-emerald-100 text-emerald-800 px-2 py-1 rounded text-sm font-medium">
                        {(data.importance * 100).toFixed(1)}%
                      </span>
                    </div>
                    
                    <div className="mb-3">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-emerald-600 h-2 rounded-full"
                          style={{ width: `${data.importance * 100}%` }}
                        ></div>
                      </div>
                    </div>
                    
                    <div className="flex flex-wrap gap-1 mb-2">
                      {data.channels.map(channel => (
                        <span key={channel} className="bg-gray-100 text-gray-700 px-2 py-1 rounded text-xs">
                          {channel}
                        </span>
                      ))}
                    </div>
                    
                    <div className="flex flex-wrap gap-1">
                      {data.emotions.map(emotion => (
                        <span key={emotion} className="bg-blue-50 text-blue-700 px-2 py-1 rounded text-xs">
                          {emotion}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            <div>
              <h3 className="font-semibold text-gray-900 mb-4">Feature Selection Impact</h3>
              <div className="bg-gray-50 rounded-lg p-6">
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-700">Original Features</span>
                    <span className="font-semibold text-gray-900">868</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-700">After Correlation Filter</span>
                    <span className="font-semibold text-gray-900">324</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-700">After Variance Filter</span>
                    <span className="font-semibold text-gray-900">187</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-700">Sequential Forward Selection</span>
                    <span className="font-semibold text-emerald-600">15</span>
                  </div>
                  <hr className="border-gray-300" />
                  <div className="flex justify-between items-center">
                    <span className="text-gray-700 font-medium">Final Accuracy</span>
                    <span className="font-bold text-emerald-600 text-lg">97.7%</span>
                  </div>
                </div>
              </div>
              
              <div className="mt-6 bg-blue-50 rounded-lg p-4">
                <h4 className="font-medium text-blue-900 mb-2">Key Insights</h4>
                <ul className="text-sm text-blue-800 space-y-1">
                  <li>• 98.3% feature reduction with accuracy gain</li>
                  <li>• Frontal channels dominate emotion recognition</li>
                  <li>• Gamma and theta bands most informative</li>
                  <li>• Cross-channel connectivity crucial</li>
                  <li>• Subject-independent performance maintained</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Feature Engineering Pipeline */}
        <div className="research-card p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            ⚙️ Feature Engineering Pipeline
          </h2>
          
          <div className="space-y-6">
            <div className="flex items-center justify-between p-4 bg-blue-50 rounded-lg">
              <div>
                <h3 className="font-semibold text-blue-900">Raw EEG Data</h3>
                <p className="text-sm text-blue-700">62 channels × 200Hz × 60s segments</p>
              </div>
              <div className="text-blue-600">↓</div>
            </div>
            
            <div className="flex items-center justify-between p-4 bg-green-50 rounded-lg">
              <div>
                <h3 className="font-semibold text-green-900">Preprocessing</h3>
                <p className="text-sm text-green-700">Bandpass filter, artifact removal, normalization</p>
              </div>
              <div className="text-green-600">↓</div>
            </div>
            
            <div className="flex items-center justify-between p-4 bg-purple-50 rounded-lg">
              <div>
                <h3 className="font-semibold text-purple-900">Feature Extraction</h3>
                <p className="text-sm text-purple-700">868 features across 4 categories</p>
              </div>
              <div className="text-purple-600">↓</div>
            </div>
            
            <div className="flex items-center justify-between p-4 bg-orange-50 rounded-lg">
              <div>
                <h3 className="font-semibold text-orange-900">Feature Selection</h3>
                <p className="text-sm text-orange-700">Sequential Forward Selection → 15 optimal features</p>
              </div>
              <div className="text-orange-600">↓</div>
            </div>
            
            <div className="flex items-center justify-between p-4 bg-emerald-50 rounded-lg">
              <div>
                <h3 className="font-semibold text-emerald-900">Model Training</h3>
                <p className="text-sm text-emerald-700">Random Forest with optimized 15 features</p>
              </div>
              <div className="text-2xl text-emerald-600">🎯</div>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-gray-900 text-white rounded-lg">
            <h4 className="font-medium mb-2">Pipeline Performance</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <div className="text-gray-300">Processing Time</div>
                <div className="font-semibold">~12 minutes</div>
              </div>
              <div>
                <div className="text-gray-300">Memory Usage</div>
                <div className="font-semibold">~8.5 GB</div>
              </div>
              <div>
                <div className="text-gray-300">Feature Reduction</div>
                <div className="font-semibold">98.3%</div>
              </div>
              <div>
                <div className="text-gray-300">Final Accuracy</div>
                <div className="font-semibold text-emerald-400">97.7%</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

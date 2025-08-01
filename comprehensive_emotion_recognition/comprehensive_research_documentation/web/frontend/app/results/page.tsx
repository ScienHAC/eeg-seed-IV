'use client';

import React, { useEffect, useState } from 'react';
import Chart from '../../components/Chart';
import PerformanceTable from '../../components/PerformanceTable';
import StageProgress from '../../components/StageProgress';

interface ResultsData {
  projectName: string;
  overallResults: {
    bestAccuracy: number;
    bestStage: number;
    bestMethod: string;
    improvementOverBaseline: number;
    totalTrainingTime: number;
  };
  stageProgression: Array<{
    stage: number;
    method: string;
    accuracy: number;
    f1Score: number;
    features: number | string;
    time: number;
    status: string;
  }>;
  emotionClassificationBreakdown: {
    stage2Results: Record<string, {
      precision: number;
      recall: number;
      f1Score: number;
      support: number;
      color: string;
    }>;
  };
  crossValidationStability: Array<{
    fold: number;
    accuracy: number;
    std: number;
  }>;
  featureDomainContribution: Array<{
    domain: string;
    contribution: number;
    keyFeatures: string;
  }>;
  downloadableReports: Array<{
    name: string;
    filename: string;
    size: string;
    description: string;
  }>;
  keyFindings: string[];
}

export default function Results() {
  const [resultsData, setResultsData] = useState<ResultsData | null>(null);

  useEffect(() => {
    fetch('/data/results.json')
      .then(res => res.json())
      .then(data => setResultsData(data))
      .catch(err => console.error('Error loading results:', err));
  }, []);

  if (!resultsData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-green-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading results...</p>
        </div>
      </div>
    );
  }

  // Prepare data for charts
  const accuracyProgressionData = resultsData.stageProgression
    .filter(stage => stage.accuracy !== null)
    .map(stage => ({
      x: `Stage ${stage.stage}`,
      y: stage.accuracy
    }));

  const performanceTableData = Object.entries(resultsData.emotionClassificationBreakdown.stage2Results).map(([emotion, metrics]) => ({
    emotion: emotion.charAt(0).toUpperCase() + emotion.slice(1),
    precision: metrics.precision,
    recall: metrics.recall,
    f1Score: metrics.f1Score,
    support: metrics.support,
    color: metrics.color
  }));

  const domainContributionData = resultsData.featureDomainContribution.map(domain => ({
    x: domain.domain,
    y: domain.contribution
  }));

  const cvStabilityData = resultsData.crossValidationStability.map(fold => ({
    x: `Fold ${fold.fold}`,
    y: fold.accuracy
  }));

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-600 to-emerald-600 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="text-center">
            <h1 className="text-4xl font-bold mb-4">
              📈 Research Results & Analysis
            </h1>
            <p className="text-xl text-green-100 mb-6">
              Comprehensive Performance Analysis of {resultsData.projectName}
            </p>
            <div className="bg-white/10 rounded-lg px-6 py-3 inline-block">
              <span className="text-2xl font-bold">
                {resultsData.overallResults.bestAccuracy}% Best Accuracy
              </span>
              <span className="text-sm ml-3">
                (+{resultsData.overallResults.improvementOverBaseline}% improvement)
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Overview Stats */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          <div className="research-card p-6 text-center">
            <div className="text-3xl font-bold text-green-600 mb-2">
              {resultsData.overallResults.bestAccuracy}%
            </div>
            <div className="text-sm text-gray-500">Best Accuracy</div>
            <div className="text-xs text-gray-400 mt-1">
              {resultsData.overallResults.bestMethod}
            </div>
          </div>
          <div className="research-card p-6 text-center">
            <div className="text-3xl font-bold text-blue-600 mb-2">
              +{resultsData.overallResults.improvementOverBaseline}%
            </div>
            <div className="text-sm text-gray-500">Improvement</div>
            <div className="text-xs text-gray-400 mt-1">
              Over baseline
            </div>
          </div>
          <div className="research-card p-6 text-center">
            <div className="text-3xl font-bold text-purple-600 mb-2">
              {resultsData.overallResults.bestStage}
            </div>
            <div className="text-sm text-gray-500">Best Stage</div>
            <div className="text-xs text-gray-400 mt-1">
              Current best
            </div>
          </div>
          <div className="research-card p-6 text-center">
            <div className="text-3xl font-bold text-orange-600 mb-2">
              {Math.round(resultsData.overallResults.totalTrainingTime / 60)}m
            </div>
            <div className="text-sm text-gray-500">Training Time</div>
            <div className="text-xs text-gray-400 mt-1">
              Total duration
            </div>
          </div>
        </div>

        {/* Stage Progress */}
        <div className="mb-12">
          <StageProgress
            currentStage={3}
            totalStages={6}
            stageNames={[
              'SVM Baseline',
              'Random Forest + SFS',
              'Autoencoder Features',
              'CNN Spatial',
              'LSTM Temporal',
              'Advanced Ensemble'
            ]}
            completedStages={[1, 2]}
          />
        </div>

        {/* Charts Grid */}
        <div className="grid md:grid-cols-2 gap-8 mb-12">
          <Chart
            data={accuracyProgressionData}
            type="line"
            title="Accuracy Progression Across Stages"
            xKey="x"
            yKey="y"
            colors={['#22c55e']}
          />
          
          <Chart
            data={domainContributionData}
            type="bar"
            title="Feature Domain Contribution"
            xKey="x"
            yKey="y"
            colors={['#3b82f6', '#10b981', '#f59e0b', '#ef4444']}
          />
        </div>

        {/* Performance Table */}
        <div className="mb-12">
          <PerformanceTable
            data={performanceTableData}
            title="Stage 2 Per-Class Performance (97.7% Accuracy)"
          />
        </div>

        {/* Cross-Validation Stability */}
        <div className="research-card p-8 mb-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            📊 Cross-Validation Stability Analysis
          </h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <Chart
                data={cvStabilityData}
                type="bar"
                title="5-Fold Cross-Validation Results"
                xKey="x"
                yKey="y"
                colors={['#6366f1']}
              />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-4">Stability Metrics</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Mean Accuracy:</span>
                  <span className="font-bold text-green-600">
                    {(resultsData.crossValidationStability.reduce((sum, fold) => sum + fold.accuracy, 0) / resultsData.crossValidationStability.length).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Standard Deviation:</span>
                  <span className="font-medium">
                    ±{resultsData.crossValidationStability[0].std}%
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Min Accuracy:</span>
                  <span className="font-medium">
                    {Math.min(...resultsData.crossValidationStability.map(f => f.accuracy)).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Max Accuracy:</span>
                  <span className="font-medium">
                    {Math.max(...resultsData.crossValidationStability.map(f => f.accuracy)).toFixed(2)}%
                  </span>
                </div>
                <div className="bg-green-50 rounded-lg p-3 mt-4">
                  <p className="text-sm text-green-800">
                    <strong>Excellent stability</strong> with low variance across folds, 
                    indicating robust model performance and good generalization.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Key Findings */}
        <div className="research-card p-8 mb-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            🔍 Key Research Findings
          </h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="font-semibold text-gray-900 mb-4">Major Breakthroughs</h3>
              <ul className="space-y-3">
                {resultsData.keyFindings.slice(0, Math.ceil(resultsData.keyFindings.length / 2)).map((finding, index) => (
                  <li key={index} className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></div>
                    <span className="text-gray-700">{finding}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-4">Technical Insights</h3>
              <ul className="space-y-3">
                {resultsData.keyFindings.slice(Math.ceil(resultsData.keyFindings.length / 2)).map((finding, index) => (
                  <li key={index} className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                    <span className="text-gray-700">{finding}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Feature Domain Analysis */}
        <div className="research-card p-8 mb-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            🔬 Feature Domain Analysis
          </h2>
          <div className="grid gap-4">
            {resultsData.featureDomainContribution.map((domain, index) => (
              <div key={index} className="flex items-center p-4 rounded-lg bg-gray-50">
                <div className="w-16 text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {domain.contribution}%
                  </div>
                </div>
                <div className="flex-1 ml-4">
                  <h3 className="font-semibold text-gray-900">{domain.domain} Domain</h3>
                  <p className="text-sm text-gray-600 mt-1">{domain.keyFeatures}</p>
                </div>
                <div className="w-32">
                  <div className="bg-gray-200 rounded-full h-3">
                    <div
                      className="bg-blue-500 h-3 rounded-full"
                      style={{ width: `${domain.contribution}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Downloadable Reports */}
        <div className="research-card p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            📥 Downloadable Reports & Data
          </h2>
          <div className="grid md:grid-cols-2 gap-6">
            {resultsData.downloadableReports.map((report, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold text-gray-900">{report.name}</h3>
                  <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                    {report.size}
                  </span>
                </div>
                <p className="text-sm text-gray-600 mb-3">{report.description}</p>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono text-gray-500">{report.filename}</span>
                  <button className="btn-primary text-xs py-1 px-3">
                    Download
                  </button>
                </div>
              </div>
            ))}
          </div>
          
          <div className="mt-6 p-4 bg-blue-50 rounded-lg">
            <p className="text-sm text-blue-800">
              <strong>Note:</strong> All results are reproducible using the provided code and configuration. 
              See the Documentation section for complete implementation details.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
